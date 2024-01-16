#![no_std]
#![deny(unused_crate_dependencies)]

#[cfg(any(feature = "std", test))]
#[macro_use]
extern crate std;

#[cfg(all(not(feature = "std"), not(test)))]
#[macro_use]
extern crate alloc;

#[cfg(all(not(feature = "std"), not(test)))]
extern crate core;

#[cfg(feature = "std")]
use std::error::Error;
#[cfg(any(feature = "std", test))]
use std::{
    fmt::{Debug, Display, Formatter, Result as FmtResult},
    marker::PhantomData,
    string::String,
    vec::Vec,
};

#[cfg(all(not(feature = "std"), not(test)))]
use alloc::{string::String, vec::Vec};
#[cfg(all(not(feature = "std"), not(test)))]
use core::{
    fmt::{Debug, Display, Formatter, Result as FmtResult},
    marker::PhantomData,
};

use external_memory_tools::ExternalMemory;

/// Trait describing how Merkle tree leaves are generated and merged.
///
/// Leaves are always sized, generic parameter N is leaf size.
pub trait Hasher<const N: usize> {
    fn make(bytes: &[u8]) -> [u8; N];
    fn merge(left: &[u8; N], right: &[u8; N]) -> [u8; N];
}

/// Trait describing how Merkle tree leaves are accessed from memory.
///
/// Need to write the leaf into memory exists only when generating proof, is
/// unlocked with feature `proof-gen`.
pub trait Leaf<const N: usize, E: ExternalMemory>: Clone + Copy + Debug + Sized {
    fn read(&self, ext_memory: &mut E) -> Result<[u8; N], E::ExternalMemoryError>;

    #[cfg(any(feature = "proof-gen", test))]
    fn write(data: [u8; N], ext_memory: &mut E) -> Result<Self, E::ExternalMemoryError>;
}

pub fn first_leaf_index(total_leaves: usize) -> usize {
    total_leaves - 1
}

pub fn number_of_layers<const N: usize, E: ExternalMemory>(
    leaves: &[Node<N>],
) -> Result<usize, ErrorMT<E>> {
    match leaves.last() {
        Some(a) => Ok(a.index.layer() as usize + 1usize),
        None => Err(ErrorMT::NoLeavesInput),
    }
}

#[cfg(any(feature = "proof-gen", test))]
fn has_duplicates<const N: usize, E: ExternalMemory>(
    values: &[[u8; N]],
) -> Result<bool, ErrorMT<E>> {
    if values.is_empty() {
        Err(ErrorMT::NoValuesInput)
    } else {
        Ok((1..values.len()).any(|i| values[i..].contains(&values[i - 1])))
    }
}

#[derive(Debug)]
pub struct MerkleProof<const N: usize, L, E, H>
where
    L: Leaf<N, E>,
    E: ExternalMemory,
    H: Hasher<N>,
{
    pub leaves: Vec<Node<N>>,
    pub lemmas: Vec<L>,
    buffer: Vec<Option<[u8; N]>>,
    previous_path: Option<Vec<Index>>,
    leftmost_leaf_position: usize,
    ext_memory: PhantomData<E>,
    hasher_type: PhantomData<H>,
}

impl<const N: usize, L, E, H> MerkleProof<N, L, E, H>
where
    L: Leaf<N, E>,
    E: ExternalMemory,
    H: Hasher<N>,
{
    pub fn new(leaves: Vec<Node<N>>, lemmas: Vec<L>) -> Result<Self, ErrorMT<E>> {
        let number_of_layers = number_of_layers(&leaves)?;

        for i in 0..leaves.len() - 1 {
            for j in i + 1..leaves.len() {
                if leaves[j].value == leaves[i].value {
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
            ext_memory: PhantomData,
            hasher_type: PhantomData,
        })
    }

    pub fn new_with_external_indices(
        leaves_values: Vec<[u8; N]>,
        indices: Vec<u32>,
        lemmas: Vec<L>,
    ) -> Result<Self, ErrorMT<E>> {
        if leaves_values.len() != indices.len() {
            return Err(ErrorMT::IndicesValuesLengthMismatch);
        }
        let mut leaves: Vec<Node<N>> = Vec::new();
        for i in 0..leaves_values.len() {
            let leaf = Node::new(Index(indices[i]), leaves_values[i]);
            leaves.push(leaf);
        }
        Self::new(leaves, lemmas)
    }

    #[cfg(any(feature = "proof-gen", test))]
    pub fn for_leaves_subset(
        all_values: Vec<[u8; N]>,
        remaining_as_leaves: &[[u8; N]],
        ext_memory: &mut E,
    ) -> Result<Self, ErrorMT<E>> {
        if has_duplicates::<N, E>(&all_values)? {
            return Err(ErrorMT::DuplicateLeafValues);
        }
        if has_duplicates::<N, E>(remaining_as_leaves)? {
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

        let mut leaves: Vec<Node<N>> = Vec::new();
        let mut lemma_collector: Vec<Node<N>> = Vec::new();

        for (index_in_whole_set, value) in all_values.into_iter().enumerate() {
            let index = Index((index_in_whole_set + first_leaf_index) as u32);
            let node = Node::new(index, value);

            if remaining_indices_in_whole_set.contains(&index_in_whole_set) {
                leaves.push(node);
            } else {
                lemma_collector.push(node);
            }
        }

        let number_of_layers = number_of_layers::<N, E>(&lemma_collector)?;

        for layer in (0..number_of_layers).rev() {
            let mut lemmas_modified = false;
            let mut removal_set: Vec<Index> = Vec::new();
            let mut added_lemmas: Vec<Node<N>> = Vec::new();
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
                                let parent_value = H::merge(&left_node.value, &right_node.value);
                                added_lemmas.push(Node::new(parent_index, parent_value));
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
        let mut lemmas: Vec<L> = Vec::new();
        for node in lemma_collector.into_iter() {
            let lemma_to_add = L::write(node.value, ext_memory).map_err(ErrorMT::ExternalMemory)?;
            lemmas.push(lemma_to_add)
        }

        Self::new(leaves, lemmas)
    }

    pub fn indices(&self) -> Vec<u32> {
        let mut out: Vec<u32> = Vec::new();
        for leaf in self.leaves.iter() {
            out.push(leaf.index.0);
        }
        out
    }

    pub fn lemmas(&self, ext_memory: &mut E) -> Result<Vec<[u8; N]>, ErrorMT<E>> {
        let mut out: Vec<[u8; N]> = Vec::new();
        for lemma in self.lemmas.iter() {
            out.push(lemma.read(ext_memory).map_err(ErrorMT::ExternalMemory)?);
        }
        Ok(out)
    }

    pub fn update(&mut self, ext_memory: &mut E) -> Result<(), ErrorMT<E>> {
        let leftmost_leaf = self.leftmost_leaf();
        let leftmost_leaf_index = leftmost_leaf.index;
        let leftmost_leaf_value = leftmost_leaf.value;

        let path_top_down = leftmost_leaf_index.path_top_down();
        let first_layer_below_bifurcation = self.first_layer_below_bifurcation(&path_top_down)?;

        let mut new_buffer_calc: Option<[u8; N]> = None;
        for (i, buffer_element) in self.buffer.iter_mut().enumerate().rev() {
            if i == first_layer_below_bifurcation {
                break;
            }
            if let Some(buffer_element_content) = buffer_element {
                if let Some(new_buffer_calc_content) = new_buffer_calc {
                    new_buffer_calc =
                        Some(H::merge(buffer_element_content, &new_buffer_calc_content));
                    *buffer_element = None;
                } else {
                    new_buffer_calc = Some(H::merge(
                        buffer_element_content,
                        &self
                            .lemmas
                            .remove(0)
                            .read(ext_memory)
                            .map_err(ErrorMT::ExternalMemory)?,
                    ));
                    *buffer_element = None;
                }
            } else if let Some(new_buffer_calc_content) = new_buffer_calc {
                new_buffer_calc = Some(H::merge(
                    &new_buffer_calc_content,
                    &self
                        .lemmas
                        .remove(0)
                        .read(ext_memory)
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
                        .read(ext_memory)
                        .map_err(ErrorMT::ExternalMemory)?,
                );
            }
        }

        let mut new_buffer_calc = leftmost_leaf_value;
        for (i, buffer_element) in self.buffer.iter_mut().enumerate().rev() {
            if leftmost_leaf_index.layer() >= i as u32 {
                if let Some(buffer_element_content) = buffer_element {
                    new_buffer_calc = H::merge(buffer_element_content, &new_buffer_calc);
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

    pub fn leftmost_leaf(&mut self) -> &Node<N> {
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

    pub fn calculate_root(&mut self, ext_memory: &mut E) -> Result<[u8; N], ErrorMT<E>> {
        let mut found_root: Option<[u8; N]> = None;

        // This guarantees that all leaves are consumed.
        for _i in 0..self.leaves.len() {
            self.update(ext_memory)?;
        }
        let mut new_buffer_calc: Option<[u8; N]> = None;
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
                        Some(H::merge(buffer_element_content, &new_buffer_calc_content));
                    *buffer_element = None;
                } else {
                    new_buffer_calc = Some(H::merge(
                        buffer_element_content,
                        &self
                            .lemmas
                            .remove(0)
                            .read(ext_memory)
                            .map_err(ErrorMT::ExternalMemory)?,
                    ));
                    *buffer_element = None;
                }
            } else if let Some(new_buffer_calc_content) = new_buffer_calc {
                new_buffer_calc = Some(H::merge(
                    &new_buffer_calc_content,
                    &self
                        .lemmas
                        .remove(0)
                        .read(ext_memory)
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

#[derive(Debug, Eq, PartialEq)]
pub enum ErrorMT<E: ExternalMemory> {
    DuplicateLeafValues,
    DuplicateRemainingValues,
    ExternalMemory(E::ExternalMemoryError),
    IndicesValuesLengthMismatch,
    LemmasNotEmpty,
    NoLeavesInput,
    NoValuesInput,
    PrevPathShorter,
    PrevPathIdentical,
    RootUnavailable,
    UnknownRemainingLeaf,
}

impl<E: ExternalMemory> ErrorMT<E> {
    fn error_text(&self) -> String {
        match &self {
            ErrorMT::DuplicateLeafValues => String::from(""),
            ErrorMT::DuplicateRemainingValues => String::from(""),
            ErrorMT::ExternalMemory(external_memory_error) => format!(" {external_memory_error}"),
            ErrorMT::IndicesValuesLengthMismatch => String::from(""),
            ErrorMT::LemmasNotEmpty => String::from(""),
            ErrorMT::NoLeavesInput => String::from(""),
            ErrorMT::NoValuesInput => String::from(""),
            ErrorMT::PrevPathShorter => String::from(""),
            ErrorMT::PrevPathIdentical => String::from(""),
            ErrorMT::RootUnavailable => String::from(""),
            ErrorMT::UnknownRemainingLeaf => String::from(""),
        }
    }
}

impl<E: ExternalMemory> Display for ErrorMT<E> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "{}", self.error_text())
    }
}

#[cfg(feature = "std")]
impl<E: ExternalMemory> Error for ErrorMT<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        None
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Node<const N: usize> {
    pub index: Index,
    pub value: [u8; N],
}

impl<const N: usize> Node<N> {
    pub fn new(index: Index, value: [u8; N]) -> Self {
        Self { index, value }
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

    const LEN: usize = 32;

    #[derive(Debug)]
    struct Blake3Hasher;

    impl Hasher<LEN> for Blake3Hasher {
        fn make(bytes: &[u8]) -> [u8; LEN] {
            blake3::hash(bytes).into()
        }
        fn merge(left: &[u8; LEN], right: &[u8; LEN]) -> [u8; LEN] {
            blake3::hash(&[left.as_slice(), right.as_slice()].concat()).into()
        }
    }

    #[derive(Copy, Clone, Debug)]
    struct Blake3Leaf([u8; LEN]);

    impl<E: ExternalMemory> Leaf<LEN, E> for Blake3Leaf {
        fn read(&self, _ext_memory: &mut E) -> Result<[u8; LEN], E::ExternalMemoryError> {
            Ok(self.0)
        }
        fn write(value: [u8; LEN], _ext_memory: &mut E) -> Result<Self, E::ExternalMemoryError> {
            Ok(Self(value))
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
        let mut merkle_proof_mock = MerkleProof::<LEN, Blake3Leaf, (), Blake3Hasher>::new(
            vec![
                Node::<LEN>::new(Index(3), [1; LEN]),
                Node::<LEN>::new(Index(9), [4; LEN]),
                Node::<LEN>::new(Index(12), [7; LEN]),
            ],
            vec![],
        )
        .unwrap();
        assert_eq!(merkle_proof_mock.leftmost_leaf().index.0, 9);

        let mut merkle_proof_mock = MerkleProof::<LEN, Blake3Leaf, (), Blake3Hasher>::new(
            vec![
                Node::<LEN>::new(Index(1), [1; LEN]),
                Node::<LEN>::new(Index(5), [6; LEN]),
                Node::<LEN>::new(Index(13), [3; LEN]),
            ],
            vec![],
        )
        .unwrap();
        assert_eq!(merkle_proof_mock.leftmost_leaf().index.0, 13);

        let mut merkle_proof_mock = MerkleProof::<LEN, Blake3Leaf, (), Blake3Hasher>::new(
            vec![
                Node::<LEN>::new(Index(1), [4; LEN]),
                Node::<LEN>::new(Index(12), [8; LEN]),
                Node::<LEN>::new(Index(6), [3; LEN]),
            ],
            vec![],
        )
        .unwrap();
        assert_eq!(merkle_proof_mock.leftmost_leaf().index.0, 12);
    }

    #[test]
    fn two_leaf_tree() {
        let mut merkle_proof = MerkleProof::<LEN, Blake3Leaf, (), Blake3Hasher>::new(
            vec![
                Node::<LEN>::new(Index(1), [0; LEN]),
                Node::<LEN>::new(Index(2), [1; LEN]),
            ],
            vec![],
        )
        .unwrap();
        let root = merkle_proof.calculate_root(&mut ()).unwrap();

        let leaves = &[[0; 32], [1; 32]];
        let root_merkle_cbt = CBMT::<[u8; 32], MergeHashes>::build_merkle_root(leaves);
        assert_eq!(root, root_merkle_cbt);
    }

    fn test_set(total_leaves: usize) -> Vec<[u8; LEN]> {
        let mut testbed: Vec<[u8; LEN]> = Vec::new();
        for i in 0..total_leaves {
            testbed.push(blake3::hash(&i.to_le_bytes()).into());
        }
        testbed
    }

    fn set_test(total_leaves: usize) {
        let test_set = test_set(total_leaves);
        let mut merkle_proof = MerkleProof::<LEN, Blake3Leaf, (), Blake3Hasher>::new(
            test_set
                .iter()
                .enumerate()
                .map(|(i, array)| {
                    Node::<LEN>::new(Index((i + first_leaf_index(total_leaves)) as u32), *array)
                })
                .collect(),
            vec![],
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
        let mut merkle_proof: MerkleProof<LEN, Blake3Leaf, (), Blake3Hasher> =
            MerkleProof::for_leaves_subset(all_values, remaining_as_leaves, &mut ()).unwrap();
        let root: [u8; 32] = merkle_proof.calculate_root(&mut ()).unwrap();

        let leaves = &[[0; 32], [1; 32]];
        let root_merkle_cbt = CBMT::<[u8; 32], MergeHashes>::build_merkle_root(leaves);
        assert_eq!(root, root_merkle_cbt);
    }

    #[test]
    fn proof_test_2() {
        let all_values = vec![[0; 32], [1; 32], [2; 32], [3; 32], [4; 32]];
        let remaining_as_leaves = &[[0; 32], [2; 32]];
        let mut merkle_proof: MerkleProof<LEN, Blake3Leaf, (), Blake3Hasher> =
            MerkleProof::for_leaves_subset(all_values, remaining_as_leaves, &mut ()).unwrap();
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
        let mut merkle_proof: MerkleProof<LEN, Blake3Leaf, (), Blake3Hasher> =
            MerkleProof::for_leaves_subset(all_values, remaining_as_leaves, &mut ()).unwrap();
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

        let mut merkle_proof: MerkleProof<LEN, Blake3Leaf, (), Blake3Hasher> =
            MerkleProof::for_leaves_subset(test_set, test_subset.as_slice(), &mut ()).unwrap();
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
            MerkleProof::<LEN, Blake3Leaf, (), Blake3Hasher>::for_leaves_subset(
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
            MerkleProof::<LEN, Blake3Leaf, (), Blake3Hasher>::for_leaves_subset(
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
            MerkleProof::<LEN, Blake3Leaf, (), Blake3Hasher>::for_leaves_subset(
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
            MerkleProof::<LEN, Blake3Leaf, (), Blake3Hasher>::new(vec![], vec![]).unwrap_err(),
            ErrorMT::NoLeavesInput
        );
    }

    #[test]
    fn error_gen_5() {
        assert_eq!(
            MerkleProof::<LEN, Blake3Leaf, (), Blake3Hasher>::for_leaves_subset(
                vec![],
                &[],
                &mut ()
            )
            .unwrap_err(),
            ErrorMT::NoValuesInput
        );
    }
}
