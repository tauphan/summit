from ...Methods.LateFusion import LateFusionClassifier
import MonoviewClassifiers
import numpy as np
import pyscm
from pyscm.utils import _pack_binary_bytes_to_ints
from utils.Dataset import getV
import os
import h5py
from pyscm.binary_attributes.classifications.popcount import inplace_popcount_32, inplace_popcount_64
from pyscm.utils import _unpack_binary_bytes_from_ints
from math import ceil


def gridSearch(DATASET, classificationKWARGS, trainIndices, nIter=30, viewsIndices=None):
    return None


class SCMForLinear(LateFusionClassifier):
    def __init__(self, NB_CORES=1, **kwargs):
        LateFusionClassifier.__init__(self, kwargs['classifiersNames'], kwargs['classifiersConfigs'],
                                      NB_CORES=NB_CORES)
        self.SCMClassifier = None

    def fit_hdf5(self, DATASET, trainIndices=None, viewsIndices=None):
        if type(viewsIndices)==type(None):
            viewsIndices = np.arange(DATASET.get("Metadata").attrs["nbView"])
        if trainIndices == None:
            trainIndices = range(DATASET.get("Metadata").attrs["datasetLength"])
        for index, viewIndex in enumerate(viewsIndices):
            monoviewClassifier = getattr(MonoviewClassifiers, self.monoviewClassifiersNames[index])
            self.monoviewClassifiers.append(
                monoviewClassifier.fit(getV(DATASET, viewIndex, trainIndices),
                                       DATASET.get("Labels")[trainIndices],
                                       NB_CORES=self.nbCores,
                                       **dict((str(configIndex), config) for configIndex, config in
                                              enumerate(self.monoviewClassifiersConfigs[index]))))
        self.SCMForLinearFusionFit(DATASET, usedIndices=trainIndices, viewsIndices=viewsIndices)

    def predict_hdf5(self, DATASET, usedIndices=None, viewsIndices=None):
        if type(viewsIndices)==type(None):
            viewsIndices = np.arange(DATASET.get("Metadata").attrs["nbView"])
        nbView = len(viewsIndices)
        if usedIndices == None:
            usedIndices = range(DATASET.get("Metadata").attrs["datasetLength"])
        if usedIndices:
            monoviewDecisions = np.zeros((len(usedIndices), nbView), dtype=int)
            for index, viewIndex in enumerate(viewsIndices):
                monoviewDecisions[:, index] = self.monoviewClassifiers[index].predict(
                    getV(DATASET, viewIndex, usedIndices))
            predictedLabels = self.SCMClassifier.predict(monoviewDecisions)
        else:
            predictedLabels = []
        return predictedLabels

    def SCMForLinearFusionFit(self, DATASET, usedIndices=None, viewsIndices=None):
        if type(viewsIndices)==type(None):
            viewsIndices = np.arange(DATASET.get("Metadata").attrs["nbView"])
        nbView = len(viewsIndices)
        self.SCMClassifier = pyscm.scm.SetCoveringMachine(p=1.0, max_attributes=5, model_type="conjunction", verbose=False)
        monoViewDecisions = np.zeros((len(usedIndices), nbView), dtype=int)
        for index, viewIndex in enumerate(viewsIndices):
            monoViewDecisions[:, index] = self.monoviewClassifiers[index].predict(
                getV(DATASET, viewIndex, usedIndices))
        featureSequence = [str(featureIndex) for featureIndex in range(monoViewDecisions.shape[1])]
        featureIndexByRule = np.arange(monoViewDecisions.shape[1], dtype=np.uint32)
        binaryAttributes = LazyBaptisteRuleList(featureSequence, featureIndexByRule)
        packedData = _pack_binary_bytes_to_ints(monoViewDecisions, 64)
        nameb = "temp_scm"
        if not os.path.isfile(nameb):
            dsetFile = h5py.File(nameb, "w")
            name=nameb
        else:
            fail=True
            i=0
            name=nameb
            while fail:
                if not os.path.isfile(name):
                    dsetFile = h5py.File(name, "w")
                    fail=False
                else:
                    i+=1
                    name = nameb+str(i)

        packedDataset = dsetFile.create_dataset("temp_scm", data=packedData)
        dsetFile.close()
        dsetFile = h5py.File(name, "r")
        packedDataset = dsetFile.get("temp_scm")
        attributeClassification = BaptisteRuleClassifications(packedDataset, monoViewDecisions.shape[0])
        self.SCMClassifier.fit(binaryAttributes, DATASET.get("Labels")[usedIndices], attribute_classifications=attributeClassification)

    def getConfig(self, fusionMethodConfig, monoviewClassifiersNames,monoviewClassifiersConfigs):
        configString = "with SCM for linear and "+str(self.SCMClassifier.attribute_importances)+" \n\t-With monoview classifiers : "
        for monoviewClassifierConfig, monoviewClassifierName in zip(monoviewClassifiersConfigs, monoviewClassifiersNames):
            monoviewClassifierModule = getattr(MonoviewClassifiers, monoviewClassifierName)
            configString += monoviewClassifierModule.getConfig(monoviewClassifierConfig)
        return configString

def _minimum_uint_size(max_value):
    """
    Find the minimum size unsigned integer type that can store values of at most max_value
    From A.Drouin's Kover
    """
    if max_value <= np.iinfo(np.uint8).max:
        return np.uint8
    elif max_value <= np.iinfo(np.uint16).max:
        return np.uint16
    elif max_value <= np.iinfo(np.uint32).max:
        return np.uint32
    elif max_value <= np.iinfo(np.uint64).max:
        return np.uint64
    else:
        return np.uint128


class BaptisteRule(object):

    def __init__(self, feature_index, kmer_sequence, type):
        """
        A k-mer rule
        Parameters:
        -----------
        feature_index: uint
            The index of the k-mer
        kmer_sequence: string
            The nucleotide sequence of the k-mer
        type: string
            The type of rule: presence or absence (use p or a)
        """
        self.feature_index = feature_index
        self.kmer_sequence = kmer_sequence
        self.type = type

    def classify(self, X):
        if self.type == "absence":
            return (X[:, self.feature_index] == 0).astype(np.uint8)
        else:
            return (X[:, self.feature_index] == 1).astype(np.uint8)

    def inverse(self):
        return BaptisteRule(feature_index=self.feature_index, kmer_sequence=self.kmer_sequence, type="absence" if self.type == "presence" else "presence")

    def __str__(self):
        return ("Absence(" if self.type == "absence" else "Presence(") + self.kmer_sequence + ")"


class LazyBaptisteRuleList(object):
    """
    By convention, the first half of the list contains presence rules and the second half contains the absence rules in
    the same order.
    """
    def __init__(self, kmer_sequences, feature_index_by_rule):
        self.n_rules = feature_index_by_rule.shape[0] * 2
        self.kmer_sequences = kmer_sequences
        self.feature_index_by_rule = feature_index_by_rule
        super(LazyBaptisteRuleList, self).__init__()

    def __getitem__(self, idx):
        if idx >= self.n_rules:
            raise ValueError("Index %d is out of range for list of size %d" % (idx, self.n_rules))
        if idx >= len(self.kmer_sequences):
            type = "absence"
            feature_idx = self.feature_index_by_rule[idx % len(self.kmer_sequences)]
        else:
            type = "presence"
            feature_idx = self.feature_index_by_rule[idx]
        return BaptisteRule(idx % len(self.kmer_sequences), self.kmer_sequences[feature_idx], type)

    def __len__(self):
        return self.n_rules

class BaseRuleClassifications(object):
    def __init__(self):
        pass

    def get_columns(self, columns):
        raise NotImplementedError()

    def remove_rows(self, rows):
        raise NotImplementedError()

    @property
    def shape(self):
        raise NotImplementedError()

    def sum_rows(self, rows):
        raise NotImplementedError()

class BaptisteRuleClassifications(BaseRuleClassifications):
    """
    Methods involving columns account for presence and absence rules
    """
    # TODO: Clean up. Get rid of the code to handle deleted rows. We don't need this.
    def __init__(self, dataset, n_rows, block_size=None):
        self.dataset = dataset
        self.dataset_initial_n_rows = n_rows
        self.dataset_n_rows = n_rows
        self.dataset_removed_rows = []
        self.dataset_removed_rows_mask = np.zeros(self.dataset_initial_n_rows, dtype=np.bool)
        self.block_size = (None, None)

        if block_size is None:
            if self.dataset.chunks is None:
                self.block_size = (1, self.dataset.shape[1])
            else:
                self.block_size = self.dataset.chunks
        else:
            if len(block_size) != 2 or not isinstance(block_size[0], int) or not isinstance(block_size[1], int):
                raise ValueError("The block size must be a tuple of 2 integers.")
            self.block_size = block_size

        # Get the size of the ints used to store the data
        if self.dataset.dtype == np.uint32:
            self.dataset_pack_size = 32
            self.inplace_popcount = inplace_popcount_32
        elif self.dataset.dtype == np.uint64:
            self.dataset_pack_size = 64
            self.inplace_popcount = inplace_popcount_64
        else:
            raise ValueError("Unsupported data type for packed attribute classifications array. The supported data" +
                             " types are np.uint32 and np.uint64.")

        super(BaseRuleClassifications, self).__init__()

    def get_columns(self, columns):
        """
        Columns can be an integer (or any object that implements __index__) or a sorted list/ndarray.
        """
        #TODO: Support slicing, make this more efficient than getting the columns individually.
        columns_is_int = False
        if hasattr(columns, "__index__"):  # All int types implement the __index__ method (PEP 357)
            columns = [columns.__index__()]
            columns_is_int = True
        elif isinstance(columns, np.ndarray):
            columns = columns.tolist()
        elif isinstance(columns, list):
            pass
        else:
            columns = list(columns)
        # Detect where an inversion is needed (columns corresponding to absence rules)
        columns, invert_result = zip(* (((column if column < self.dataset.shape[1] else column % self.dataset.shape[1]),
                                         (True if column > self.dataset.shape[1] else False)) for column in columns))
        columns = list(columns)
        invert_result = np.array(invert_result)

        # Don't return rows that have been deleted
        row_mask = np.ones(self.dataset.shape[0] * self.dataset_pack_size, dtype=np.bool)
        row_mask[self.dataset_initial_n_rows:] = False
        row_mask[self.dataset_removed_rows] = False

        # h5py requires that the column indices are sorted
        unique, inverse = np.unique(columns, return_inverse=True)
        result = _unpack_binary_bytes_from_ints(self.dataset[:, unique.tolist()])[row_mask]
        result = result[:, inverse]
        result[:, invert_result] = 1 - result[:, invert_result]

        if columns_is_int:
            return result.reshape(-1)
        else:
            return result

    @property
    def shape(self):
        return self.dataset_n_rows, self.dataset.shape[1] * 2

    # TODO: allow summing over multiple lists of rows at a time (saves i/o operations)
    def sum_rows(self, rows):
        """
        Note: Assumes that the rows argument does not contain duplicate elements. Rows will not be considered more than once.
        """
        rows = np.asarray(rows)
        result_dtype = _minimum_uint_size(rows.shape[0])
        result = np.zeros(self.dataset.shape[1] * 2, dtype=result_dtype)

        # Builds a mask to turn off the bits of the rows we do not want to count in the sum.
        def build_row_mask(example_idx, n_examples, mask_n_bits):
            if mask_n_bits not in [8, 16, 32, 64, 128]:
                raise ValueError("Unsupported mask format. Use 8, 16, 32, 64 or 128 bits.")

            n_masks = int(ceil(float(n_examples) / mask_n_bits))
            masks = [0] * n_masks

            for idx in example_idx:
                example_mask = idx / mask_n_bits
                example_mask_idx = mask_n_bits - (idx - mask_n_bits * example_mask) - 1
                masks[example_mask] |= 1 << example_mask_idx

            return np.array(masks, dtype="u" + str(mask_n_bits / 8))

        # Find the rows that occur in each dataset and their relative index
        rows = np.sort(rows)
        dataset_relative_rows = []
        for row_idx in rows:
            # Find which row in the dataset corresponds to the requested row
            # TODO: This is inefficient! Could exploit the fact that rows is sorted to reuse previous iterations.
            current_idx = -1
            n_active_elements_seen = 0
            while n_active_elements_seen <= row_idx:
                current_idx += 1
                if not self.dataset_removed_rows_mask[current_idx]:
                    n_active_elements_seen += 1
            dataset_relative_rows.append(current_idx)

        # Create a row mask for each dataset
        row_mask = build_row_mask(dataset_relative_rows, self.dataset_initial_n_rows, self.dataset_pack_size)
        del dataset_relative_rows

        # For each dataset load the rows for which the mask is not 0. Support column slicing aswell
        n_col_blocks = int(ceil(1.0 * self.dataset.shape[1] / self.block_size[1]))
        rows_to_load = np.where(row_mask != 0)[0]
        n_row_blocks = int(ceil(1.0 * len(rows_to_load) / self.block_size[0]))

        for row_block in xrange(n_row_blocks):
            block_row_mask = row_mask[rows_to_load[row_block * self.block_size[0]:(row_block + 1) * self.block_size[0]]]

            for col_block in xrange(n_col_blocks):

                # Load the appropriate rows/columns based on the block sizes
                block = self.dataset[rows_to_load[row_block * self.block_size[0]:(row_block + 1) * self.block_size[0]],
                        col_block * self.block_size[1]:(col_block + 1) * self.block_size[1]]

                # Popcount
                if len(block.shape) == 1:
                    block = block.reshape(1, -1)
                self.inplace_popcount(block, block_row_mask)

                # Increment the sum
                result[col_block * self.block_size[1]:min((col_block + 1) * self.block_size[1], self.dataset.shape[1])] += np.sum(block, axis=0)

        # Compute the sum for absence rules
        result[self.dataset.shape[1] : ] = len(rows) - result[: self.dataset.shape[1]]

        return result