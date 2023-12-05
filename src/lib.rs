#![no_std]
#![deny(unused_crate_dependencies)]

#[cfg(any(feature = "std", test))]
extern crate std;

#[cfg(all(not(feature = "std"), not(test)))]
extern crate alloc;

#[cfg(all(not(feature = "std"), not(test)))]
extern crate core;

#[cfg(any(feature = "std", test))]
use std::{
    fmt::{Debug, Display, Formatter, Result as FmtResult},
    marker::PhantomData,
    vec::Vec,
};

#[cfg(all(not(feature = "std"), not(test)))]
use alloc::vec::Vec;
#[cfg(all(not(feature = "std"), not(test)))]
use core::{
    fmt::{Debug, Display, Formatter, Result as FmtResult},
    marker::PhantomData,
};

pub trait ExternalMemory: Debug {
    type ExternalMemoryError: Debug + Display + Eq + PartialEq;
}

impl ExternalMemory for () {
    type ExternalMemoryError = NoEntries;
}

#[derive(Debug, Eq, PartialEq)]
pub enum NoEntries {}

impl Display for NoEntries {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "")
    }
}

pub trait Leaf<E: ExternalMemory>: Clone + Copy + Debug {
    type Value: AsRef<[u8]> + Clone + Copy + Debug + PartialEq;
    fn merge(left: &Self::Value, right: &Self::Value) -> Self::Value;
    fn value(&self, ext_memory: &mut E) -> Result<Self::Value, E::ExternalMemoryError>;
    fn from_value(value: Self::Value, ext_memory: &mut E) -> Result<Self, E::ExternalMemoryError>;
}

#[derive(Clone, Copy, Debug)]
pub struct Blake3Leaf([u8; 32]);

impl<E: ExternalMemory> Leaf<E> for Blake3Leaf {
    type Value = [u8; 32];
    fn merge(left: &Self::Value, right: &Self::Value) -> Self::Value {
        blake3::hash(&[left.as_slice(), right.as_slice()].concat()).into()
    }
    fn value(&self, _ext_memory: &mut E) -> Result<Self::Value, E::ExternalMemoryError> {
        Ok(self.0)
    }
    fn from_value(value: Self::Value, _ext_memory: &mut E) -> Result<Self, E::ExternalMemoryError> {
        Ok(Self(value))
    }
}

impl Blake3Leaf {
    pub fn from_unhashed(unhashed: &[u8]) -> Self {
        Self(blake3::hash(unhashed).into())
    }
}

pub fn first_leaf_index(total_leaves: usize) -> usize {
    total_leaves - 1
}

pub fn number_of_layers<L, E>(leaves: &[Node<L, E>]) -> Result<usize, ErrorMT<E>>
where
    L: Leaf<E>,
    E: ExternalMemory,
{
    match leaves.last() {
        Some(a) => Ok(a.index.layer() as usize + 1usize),
        None => Err(ErrorMT::NoLeavesInput),
    }
}

fn has_duplicates<L, E>(values: &[L::Value]) -> Result<bool, ErrorMT<E>>
where
    L: Leaf<E>,
    E: ExternalMemory,
{
    if values.is_empty() {
        Err(ErrorMT::NoValuesInput)
    } else {
        Ok((1..values.len()).any(|i| values[i..].contains(&values[i - 1])))
    }
}

#[derive(Debug)]
pub struct MerkleProof<L, E>
where
    L: Leaf<E>,
    E: ExternalMemory,
{
    pub leaves: Vec<Node<L, E>>,
    pub lemmas: Vec<L>,
    buffer: Vec<Option<L::Value>>,
    previous_path: Option<Vec<Index>>,
    leftmost_leaf_position: usize,
}

impl<L, E> MerkleProof<L, E>
where
    L: Leaf<E>,
    E: ExternalMemory,
{
    pub fn new(
        leaves: Vec<Node<L, E>>,
        lemmas: Vec<L>,
        ext_memory: &mut E,
    ) -> Result<Self, ErrorMT<E>> {
        let number_of_layers = number_of_layers(&leaves)?;

        for i in 0..leaves.len() - 1 {
            for j in i + 1..leaves.len() {
                if leaves[j].value.value(ext_memory) == leaves[i].value.value(ext_memory) {
                    return Err(ErrorMT::DuplicateLeafValues);
                }
            }
        }

        let mut buffer = Vec::with_capacity(number_of_layers);
        for _i in 0..number_of_layers {
            buffer.push(None);
        }

        let last_layer = number_of_layers - 1;
        let last_layer_leftmost_node = 2u32.pow(last_layer as u32) - 1;
        let leftmost_leaf_position = leaves
            .iter()
            .position(|leaf| leaf.index.0 >= last_layer_leftmost_node)
            .expect("last layer always contains at least a single leaf");

        Ok(Self {
            leaves,
            lemmas,
            buffer,
            previous_path: None,
            leftmost_leaf_position,
        })
    }

    pub fn for_leaves_subset(
        all_values: Vec<L::Value>,
        remaining_as_leaves: &[L::Value],
        ext_memory: &mut E,
    ) -> Result<Self, ErrorMT<E>> {
        if has_duplicates::<L, E>(&all_values)? {
            return Err(ErrorMT::DuplicateLeafValues);
        }
        if has_duplicates::<L, E>(remaining_as_leaves)? {
            return Err(ErrorMT::DuplicateRemainingValues);
        }

        let first_leaf_index = first_leaf_index(all_values.len());

        let mut remaining_indices_in_whole_set: Vec<usize> = Vec::new();

        for remaining_value in remaining_as_leaves.iter() {
            let index_in_whole_set = all_values
                .iter()
                .position(|value| value == remaining_value)
                .ok_or(ErrorMT::UnknownRemainingLeaf)?;
            remaining_indices_in_whole_set.push(index_in_whole_set);
        }

        let mut leaves: Vec<Node<L, E>> = Vec::new();
        let mut lemma_collector: Vec<Node<L, E>> = Vec::new();

        for (index_in_whole_set, value) in all_values.into_iter().enumerate() {
            let index = Index((index_in_whole_set + first_leaf_index) as u32);
            let value = L::from_value(value, ext_memory).map_err(ErrorMT::ExternalMemory)?;
            let node = Node::new(index, value);

            if remaining_indices_in_whole_set.contains(&index_in_whole_set) {
                leaves.push(node);
            } else {
                lemma_collector.push(node);
            }
        }

        let number_of_layers = number_of_layers::<L, E>(&lemma_collector)?;

        for layer in (0..number_of_layers).rev() {
            let mut lemmas_modified = false;
            let mut removal_set: Vec<Index> = Vec::new();
            let mut added_lemmas: Vec<Node<L, E>> = Vec::new();
            for (n, node) in lemma_collector.iter().enumerate() {
                if node.index.layer() == layer as u32 {
                    if let Some(node_sibling_index) = node.index.sibling() {
                        if let Some(i) = lemma_collector.iter().position(|node_in_collector| {
                            node_in_collector.index == node_sibling_index
                        }) {
                            if node.index.is_left() {
                                removal_set.push(node.index);
                                removal_set.push(node_sibling_index);
                                let left_node = &lemma_collector[n];
                                let right_node = &lemma_collector[i];
                                let parent_index = left_node
                                    .index
                                    .parent()
                                    .expect("if there is a sibling, there must be a parent");
                                let parent_value = L::merge(
                                    &left_node
                                        .value
                                        .value(ext_memory)
                                        .map_err(ErrorMT::ExternalMemory)?,
                                    &right_node
                                        .value
                                        .value(ext_memory)
                                        .map_err(ErrorMT::ExternalMemory)?,
                                );
                                added_lemmas.push(Node::new(
                                    parent_index,
                                    L::from_value(parent_value, ext_memory)
                                        .map_err(ErrorMT::ExternalMemory)?,
                                ));
                                lemmas_modified = true;
                            }
                        }
                    }
                }
            }
            lemma_collector.retain(|x| !removal_set.contains(&x.index));
            lemma_collector.append(&mut added_lemmas);
            if !lemmas_modified {
                break;
            }
        }

        lemma_collector.sort_by(|a, b| a.index.path_top_down().cmp(&b.index.path_top_down()));
        let lemmas = lemma_collector.into_iter().map(|node| node.value).collect();

        Self::new(leaves, lemmas, ext_memory)
    }

    pub fn update(&mut self, ext_memory: &mut E) -> Result<(), ErrorMT<E>> {
        let leftmost_leaf = self.leftmost_leaf();
        let leftmost_leaf_index = leftmost_leaf.index;
        let leftmost_leaf_value = leftmost_leaf
            .value
            .value(ext_memory)
            .map_err(ErrorMT::ExternalMemory)?;

        let path_top_down = leftmost_leaf_index.path_top_down();
        let first_layer_below_bifurcation = self.first_layer_below_bifurcation(&path_top_down)?;

        let mut new_buffer_calc: Option<L::Value> = None;
        for (i, buffer_element) in self.buffer.iter_mut().enumerate().rev() {
            if i == first_layer_below_bifurcation {
                break;
            }
            if let Some(buffer_element_content) = buffer_element {
                if let Some(new_buffer_calc_content) = new_buffer_calc {
                    new_buffer_calc =
                        Some(L::merge(buffer_element_content, &new_buffer_calc_content));
                    *buffer_element = None;
                } else {
                    new_buffer_calc = Some(L::merge(
                        buffer_element_content,
                        &self
                            .lemmas
                            .remove(0)
                            .value(ext_memory)
                            .map_err(ErrorMT::ExternalMemory)?,
                    ));
                    *buffer_element = None;
                }
            } else if let Some(new_buffer_calc_content) = new_buffer_calc {
                new_buffer_calc = Some(L::merge(
                    &new_buffer_calc_content,
                    &self
                        .lemmas
                        .remove(0)
                        .value(ext_memory)
                        .map_err(ErrorMT::ExternalMemory)?,
                ));
            }
        }
        if let Some(new_buffer_calc_content) = new_buffer_calc {
            self.buffer[first_layer_below_bifurcation] = Some(new_buffer_calc_content);
        }

        for (layer, index) in path_top_down.iter().enumerate() {
            if layer < first_layer_below_bifurcation {
                continue;
            }
            if !index.is_left() {
                self.buffer[layer + 1] = Some(
                    self.lemmas
                        .remove(0)
                        .value(ext_memory)
                        .map_err(ErrorMT::ExternalMemory)?,
                );
            }
        }

        let mut new_buffer_calc = leftmost_leaf_value;
        for (i, buffer_element) in self.buffer.iter_mut().enumerate().rev() {
            if leftmost_leaf_index.layer() >= i as u32 {
                if let Some(buffer_element_content) = buffer_element {
                    new_buffer_calc = L::merge(buffer_element_content, &new_buffer_calc);
                    *buffer_element = None;
                } else {
                    *buffer_element = Some(new_buffer_calc);
                    break;
                }
            }
        }
        self.previous_path = Some(path_top_down);
        Ok(())
    }

    pub fn leftmost_leaf(&mut self) -> &Node<L, E> {
        let leftmost_leaf = &self.leaves[self.leftmost_leaf_position];
        self.leftmost_leaf_position += 1;
        if self.leftmost_leaf_position == self.leaves.len() {
            self.leftmost_leaf_position = 0;
        }
        leftmost_leaf
    }

    pub fn first_layer_below_bifurcation(&self, path: &[Index]) -> Result<usize, ErrorMT<E>> {
        match &self.previous_path {
            Some(known_previous_path) => {
                for (i, path_element) in path.iter().enumerate() {
                    match known_previous_path.get(i) {
                        Some(previous_path_element) => {
                            if previous_path_element != path_element {
                                return Ok(i + 1);
                            }
                        }
                        None => return Err(ErrorMT::PrevPathShorter),
                    }
                }
                Err(ErrorMT::PrevPathIdentical)
            }
            None => Ok(0usize),
        }
    }

    pub fn calculate_root(&mut self, ext_memory: &mut E) -> Result<L::Value, ErrorMT<E>> {
        let mut found_root: Option<L::Value> = None;

        // This guarantees that all leaves are consumed.
        for _i in 0..self.leaves.len() {
            self.update(ext_memory)?;
        }
        let mut new_buffer_calc: Option<L::Value> = None;
        for (i, buffer_element) in self.buffer.iter_mut().enumerate().rev() {
            if i == 0 {
                if new_buffer_calc.is_some() {
                    found_root = new_buffer_calc
                }
                break;
            }
            if let Some(buffer_element_content) = buffer_element {
                if let Some(new_buffer_calc_content) = new_buffer_calc {
                    new_buffer_calc =
                        Some(L::merge(buffer_element_content, &new_buffer_calc_content));
                    *buffer_element = None;
                } else {
                    new_buffer_calc = Some(L::merge(
                        buffer_element_content,
                        &self
                            .lemmas
                            .remove(0)
                            .value(ext_memory)
                            .map_err(ErrorMT::ExternalMemory)?,
                    ));
                    *buffer_element = None;
                }
            } else if let Some(new_buffer_calc_content) = new_buffer_calc {
                new_buffer_calc = Some(L::merge(
                    &new_buffer_calc_content,
                    &self
                        .lemmas
                        .remove(0)
                        .value(ext_memory)
                        .map_err(ErrorMT::ExternalMemory)?,
                ));
            }
        }
        if self.buffer[0].is_some() {
            found_root = self.buffer[0];
        }
        match found_root {
            Some(root) => {
                if !self.lemmas.is_empty() {
                    return Err(ErrorMT::LemmasNotEmpty);
                }
                Ok(root)
            }
            None => Err(ErrorMT::RootUnavailable),
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum ErrorMT<E: ExternalMemory> {
    DuplicateLeafValues,
    DuplicateRemainingValues,
    ExternalMemory(E::ExternalMemoryError),
    LemmasNotEmpty,
    NoLeavesInput,
    NoValuesInput,
    PrevPathShorter,
    PrevPathIdentical,
    RootUnavailable,
    UnknownRemainingLeaf,
}

#[derive(Clone, Copy, Debug)]
pub struct Node<L, E>
where
    L: Leaf<E>,
    E: ExternalMemory,
{
    pub index: Index,
    pub value: L,
    ext_memory_type: PhantomData<E>,
}

impl<L, E> Node<L, E>
where
    L: Leaf<E>,
    E: ExternalMemory,
{
    pub fn new(index: Index, value: L) -> Self {
        Self {
            index,
            value,
            ext_memory_type: PhantomData,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct Index(pub u32);

impl Index {
    pub fn layer(&self) -> u32 {
        (self.0 + 1).ilog2()
    }

    pub fn path_top_down(&self) -> Vec<Index> {
        let mut out: Vec<Index> = Vec::with_capacity(self.layer() as usize);
        let mut current_index = *self;
        while let Some(current_parent) = current_index.parent() {
            out.push(current_index);
            current_index = current_parent;
        }
        out.reverse();
        out
    }

    pub fn sibling(&self) -> Option<Self> {
        if self.0 == 0 {
            None
        } else {
            Some(Index(((self.0 + 1) ^ 1) - 1))
        }
    }

    pub fn parent(&self) -> Option<Self> {
        if self.0 == 0 {
            None
        } else {
            Some(Index((self.0 - 1) / 2))
        }
    }

    pub fn is_left(&self) -> bool {
        self.0 % 2 == 1
    }
}

#[cfg(test)]
mod tests {
    use merkle_cbt::{merkle_tree::Merge, CBMT};
    use std::vec;

    use super::*;

    struct MergeHashes;

    impl Merge for MergeHashes {
        type Item = [u8; 32];
        fn merge(left: &Self::Item, right: &Self::Item) -> Self::Item {
            blake3::hash(&[*left, *right].concat()).into()
        }
    }

    #[test]
    fn find_node_layer() {
        assert_eq!(Index(0).layer(), 0);
        assert_eq!(Index(1).layer(), 1);
        assert_eq!(Index(2).layer(), 1);
        assert_eq!(Index(4).layer(), 2);
        assert_eq!(Index(12).layer(), 3);
    }

    #[test]
    fn node_is_left() {
        assert!(Index(1).is_left());
        assert!(!Index(2).is_left());
        assert!(!Index(4).is_left());
        assert!(Index(11).is_left());
    }

    #[test]
    fn correct_path() {
        assert_eq!(Index(1).path_top_down(), vec![Index(1)]);
        assert_eq!(Index(2).path_top_down(), vec![Index(2)]);
        assert_eq!(Index(0).path_top_down(), vec![]);
        assert_eq!(
            Index(12).path_top_down(),
            vec![Index(2), Index(5), Index(12)]
        );
        assert_eq!(
            Index(13).path_top_down(),
            vec![Index(2), Index(6), Index(13)]
        );
    }

    #[test]
    fn leftmost_node() {
        let mut merkle_proof_mock = MerkleProof::<Blake3Leaf, ()>::new(
            vec![
                Node::<Blake3Leaf, ()>::new(Index(3), Blake3Leaf([1; 32])),
                Node::<Blake3Leaf, ()>::new(Index(9), Blake3Leaf([4; 32])),
                Node::<Blake3Leaf, ()>::new(Index(12), Blake3Leaf([7; 32])),
            ],
            vec![],
            &mut (),
        )
        .unwrap();
        assert_eq!(merkle_proof_mock.leftmost_leaf().index.0, 9);

        let mut merkle_proof_mock = MerkleProof::<Blake3Leaf, ()>::new(
            vec![
                Node::<Blake3Leaf, ()>::new(Index(1), Blake3Leaf([1; 32])),
                Node::<Blake3Leaf, ()>::new(Index(5), Blake3Leaf([6; 32])),
                Node::<Blake3Leaf, ()>::new(Index(13), Blake3Leaf([3; 32])),
            ],
            vec![],
            &mut (),
        )
        .unwrap();
        assert_eq!(merkle_proof_mock.leftmost_leaf().index.0, 13);

        let mut merkle_proof_mock = MerkleProof::<Blake3Leaf, ()>::new(
            vec![
                Node::<Blake3Leaf, ()>::new(Index(1), Blake3Leaf([4; 32])),
                Node::<Blake3Leaf, ()>::new(Index(12), Blake3Leaf([8; 32])),
                Node::<Blake3Leaf, ()>::new(Index(6), Blake3Leaf([3; 32])),
            ],
            vec![],
            &mut (),
        )
        .unwrap();
        assert_eq!(merkle_proof_mock.leftmost_leaf().index.0, 12);
    }

    #[test]
    fn two_leaf_tree() {
        let mut merkle_proof = MerkleProof::<Blake3Leaf, ()>::new(
            vec![
                Node::<Blake3Leaf, ()>::new(Index(1), Blake3Leaf([0; 32])),
                Node::<Blake3Leaf, ()>::new(Index(2), Blake3Leaf([1; 32])),
            ],
            vec![],
            &mut (),
        )
        .unwrap();
        let root = merkle_proof.calculate_root(&mut ()).unwrap();

        let leaves = &[[0; 32], [1; 32]];
        let root_merkle_cbt = CBMT::<[u8; 32], MergeHashes>::build_merkle_root(leaves);
        assert_eq!(root, root_merkle_cbt);
    }

    fn test_set(total_leaves: usize) -> Vec<[u8; 32]> {
        let mut testbed: Vec<[u8; 32]> = Vec::new();
        for i in 0..total_leaves {
            testbed.push(blake3::hash(&i.to_le_bytes()).into());
        }
        testbed
    }

    fn set_test(total_leaves: usize) {
        let test_set = test_set(total_leaves);
        let mut merkle_proof = MerkleProof::<Blake3Leaf, ()>::new(
            test_set
                .iter()
                .enumerate()
                .map(|(i, array)| {
                    Node::<Blake3Leaf, ()>::new(
                        Index((i + first_leaf_index(total_leaves)) as u32),
                        Blake3Leaf(*array),
                    )
                })
                .collect(),
            vec![],
            &mut (),
        )
        .unwrap();
        let root = merkle_proof.calculate_root(&mut ()).unwrap();

        let root_merkle_cbt = CBMT::<[u8; 32], MergeHashes>::build_merkle_root(&test_set);
        assert_eq!(root, root_merkle_cbt);
    }

    #[test]
    fn set_tree_1() {
        set_test(3)
    }

    #[test]
    fn set_tree_2() {
        set_test(4)
    }

    #[test]
    fn set_tree_3() {
        set_test(15)
    }

    #[test]
    fn set_tree_4() {
        set_test(16)
    }

    #[test]
    fn set_tree_5() {
        set_test(17)
    }

    #[test]
    fn set_tree_6() {
        set_test(10000)
    }

    #[test]
    fn proof_test_1() {
        let all_values = vec![[0; 32], [1; 32]];
        let remaining_as_leaves = &[[0; 32]];
        let mut merkle_proof: MerkleProof<Blake3Leaf, ()> =
            MerkleProof::<Blake3Leaf, ()>::for_leaves_subset(
                all_values,
                remaining_as_leaves,
                &mut (),
            )
            .unwrap();
        let root: [u8; 32] = merkle_proof.calculate_root(&mut ()).unwrap();

        let leaves = &[[0; 32], [1; 32]];
        let root_merkle_cbt = CBMT::<[u8; 32], MergeHashes>::build_merkle_root(leaves);
        assert_eq!(root, root_merkle_cbt);
    }

    #[test]
    fn proof_test_2() {
        let all_values = vec![[0; 32], [1; 32], [2; 32], [3; 32], [4; 32]];
        let remaining_as_leaves = &[[0; 32], [2; 32]];
        let mut merkle_proof: MerkleProof<Blake3Leaf, ()> =
            MerkleProof::<Blake3Leaf, ()>::for_leaves_subset(
                all_values,
                remaining_as_leaves,
                &mut (),
            )
            .unwrap();
        let root: [u8; 32] = merkle_proof.calculate_root(&mut ()).unwrap();

        let leaves = &[[0; 32], [1; 32], [2; 32], [3; 32], [4; 32]];
        let root_merkle_cbt = CBMT::<[u8; 32], MergeHashes>::build_merkle_root(leaves);
        assert_eq!(root, root_merkle_cbt);
    }

    #[test]
    fn proof_test_3() {
        let all_values = vec![
            [0; 32], [1; 32], [2; 32], [3; 32], [4; 32], [5; 32], [6; 32],
        ];
        let remaining_as_leaves = &[[0; 32], [6; 32]];
        let mut merkle_proof: MerkleProof<Blake3Leaf, ()> =
            MerkleProof::<Blake3Leaf, ()>::for_leaves_subset(
                all_values,
                remaining_as_leaves,
                &mut (),
            )
            .unwrap();
        let root: [u8; 32] = merkle_proof.calculate_root(&mut ()).unwrap();

        let leaves = &[
            [0; 32], [1; 32], [2; 32], [3; 32], [4; 32], [5; 32], [6; 32],
        ];
        let root_merkle_cbt = CBMT::<[u8; 32], MergeHashes>::build_merkle_root(leaves);
        assert_eq!(root, root_merkle_cbt);
    }

    #[test]
    fn proof_test_4() {
        let total_leaves = 1000;
        let test_set = test_set(total_leaves);
        let test_subset = [test_set[1], test_set[12], test_set[118]];
        let root_merkle_cbt = CBMT::<[u8; 32], MergeHashes>::build_merkle_root(&test_set);

        let mut merkle_proof = MerkleProof::<Blake3Leaf, ()>::for_leaves_subset(
            test_set,
            test_subset.as_slice(),
            &mut (),
        )
        .unwrap();
        let root: [u8; 32] = merkle_proof.calculate_root(&mut ()).unwrap();

        assert_eq!(root, root_merkle_cbt);
    }

    #[test]
    fn error_gen_1() {
        let all_values = vec![
            [0; 32], [1; 32], [2; 32], [3; 32], [4; 32], [5; 32], [0; 32],
        ];
        let remaining_as_leaves = &[[0; 32], [3; 32]];
        assert_eq!(
            MerkleProof::<Blake3Leaf, ()>::for_leaves_subset(
                all_values,
                remaining_as_leaves,
                &mut ()
            )
            .unwrap_err(),
            ErrorMT::DuplicateLeafValues
        );
    }

    #[test]
    fn error_gen_2() {
        let all_values = vec![[0; 32], [1; 32], [2; 32], [3; 32]];
        let remaining_as_leaves = &[[1; 32], [1; 32]];
        assert_eq!(
            MerkleProof::<Blake3Leaf, ()>::for_leaves_subset(
                all_values,
                remaining_as_leaves,
                &mut ()
            )
            .unwrap_err(),
            ErrorMT::DuplicateRemainingValues
        );
    }

    #[test]
    fn error_gen_3() {
        let all_values = vec![[0; 32], [1; 32], [2; 32], [3; 32], [4; 32], [5; 32]];
        let remaining_as_leaves = &[[0; 32], [6; 32]];
        assert_eq!(
            MerkleProof::<Blake3Leaf, ()>::for_leaves_subset(
                all_values,
                remaining_as_leaves,
                &mut ()
            )
            .unwrap_err(),
            ErrorMT::UnknownRemainingLeaf
        );
    }

    #[test]
    fn error_gen_4() {
        assert_eq!(
            MerkleProof::<Blake3Leaf, ()>::new(vec![], vec![], &mut ()).unwrap_err(),
            ErrorMT::NoLeavesInput
        );
    }

    #[test]
    fn error_gen_5() {
        assert_eq!(
            MerkleProof::<Blake3Leaf, ()>::for_leaves_subset(vec![], &[], &mut ()).unwrap_err(),
            ErrorMT::NoValuesInput
        );
    }
}
