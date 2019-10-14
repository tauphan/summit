import logging
import os
import select
import sys
import errno

import h5py
import numpy as np
from scipy import sparse

# from . import get_multiview_db as DB


class Dataset():

    # The following methods use hdf5

    def __init__(self, views=None, labels=None, are_sparse=False,
                 file_name="dataset.hdf5", view_names=None, path="",
                 hdf5_file=None, labels_names=None, is_temp=False):
        self.is_temp = False
        if hdf5_file is not None:
            self.dataset=hdf5_file
            self.init_attrs()
        else:
            if not os.path.exists(os.path.dirname(os.path.join(path, file_name))):
                try:
                    os.makedirs(os.path.dirname(os.path.join(path, file_name)))
                except OSError as exc:
                    if exc.errno != errno.EEXIST:
                        raise
            dataset_file = h5py.File(os.path.join(path, file_name), "w")
            if view_names is None:
                view_names = ["View"+str(index) for index in range(len(views))]
            if isinstance(are_sparse, bool):
                are_sparse = [are_sparse for _ in views]
            for view_index, (view_name, view, is_sparse) in enumerate(zip(view_names, views, are_sparse)):
                view_dataset = dataset_file.create_dataset("View" + str(view_index),
                                                      view.shape,
                                                      data=view)
                view_dataset.attrs["name"] = view_name
                view_dataset.attrs["sparse"] = is_sparse
            labels_dataset = dataset_file.create_dataset("Labels",
                                                         shape=labels.shape,
                                                         data=labels)
            if labels_names is None:
                labels_names = [str(index) for index in np.unique(labels)]
            labels_dataset.attrs["names"] = [label_name.encode()
                                             if not isinstance(label_name, bytes)
                                             else label_name
                                             for label_name in labels_names]
            meta_data_grp = dataset_file.create_group("Metadata")
            meta_data_grp.attrs["nbView"] = len(views)
            meta_data_grp.attrs["nbClass"] = len(np.unique(labels))
            meta_data_grp.attrs["datasetLength"] = len(labels)
            dataset_file.close()
            self.update_hdf5_dataset(os.path.join(path, file_name))

    def rm(self):
        filename = self.dataset.filename
        self.dataset.close()
        if self.is_temp:
            os.remove(filename)

    def get_view_name(self, view_idx):
        return self.dataset["View"+str(view_idx)].attrs["name"]

    def init_attrs(self):
        self.nb_view = self.dataset.get("Metadata").attrs["nbView"]
        self.view_dict = self.get_view_dict()

    def get_nb_examples(self):
        return self.dataset.get("Metadata").attrs["datasetLength"]

    def get_view_dict(self):
        view_dict = {}
        for view_index in range(self.nb_view):
            view_dict[self.dataset.get("View" + str(view_index)).attrs["name"]] = view_index
        return view_dict

    def get_label_names(self, decode=True, example_indices=None):
        example_indices = self.init_example_indces(example_indices)
        selected_labels = self.get_labels(example_indices)
        if decode:
            return [label_name.decode("utf-8")
                    for label, label_name in enumerate(self.dataset.get("Labels").attrs["names"])
                    if label in selected_labels]
        else:
            return [label_name
                    for label, label_name in enumerate(self.dataset.get("Labels").attrs["names"])
                    if label in selected_labels]

    def init_example_indces(self, example_indices=None):
        if example_indices is None:
            return range(self.get_nb_examples())
        else:
            return example_indices

    def get_v(self, view_index, example_indices=None):
        example_indices = self.init_example_indces(example_indices)
        if type(example_indices) is int:
            return self.dataset.get("View" + str(view_index))[example_indices, :]
        else:
            example_indices = np.array(example_indices)
            sorted_indices = np.argsort(example_indices)
            example_indices = example_indices[sorted_indices]

            if not self.dataset.get("View" + str(view_index)).attrs["sparse"]:
                return self.dataset.get("View" + str(view_index))[example_indices, :][
                       np.argsort(sorted_indices), :]
            else:
                sparse_mat = sparse.csr_matrix(
                    (self.dataset.get("View" + str(view_index)).get("data").value,
                     self.dataset.get("View" + str(view_index)).get("indices").value,
                     self.dataset.get("View" + str(view_index)).get("indptr").value),
                    shape=self.dataset.get("View" + str(view_index)).attrs["shape"])[
                             example_indices, :][
                             np.argsort(sorted_indices), :]

                return sparse_mat

    def get_shape(self, example_indices=None):
        return self.get_v(0,example_indices=example_indices).shape

    def get_nb_class(self, example_indices=None):
        example_indices = self.init_example_indces(example_indices)
        return len(np.unique(self.dataset.get("Labels").value[example_indices]))

    def get_labels(self, example_indices=None):
        example_indices = self.init_example_indces(example_indices)
        return self.dataset.get("Labels").value[example_indices]

    def copy_view(self, target_dataset=None, source_view_name=None,
                  target_view_index=None, example_indices=None):
        example_indices = self.init_example_indces(example_indices)
        new_d_set = target_dataset.create_dataset("View"+str(target_view_index),
            data=self.get_v(self.view_dict[source_view_name],
                            example_indices=example_indices))
        for key, value in self.dataset.get("View"+str(self.view_dict[source_view_name])).attrs.items():
            new_d_set.attrs[key] = value

    def init_view_names(self, view_names=None):
        if view_names is None:
            return [key for key in self.get_view_dict().keys()]
        else:
            return view_names

    def update_hdf5_dataset(self, path):
        if hasattr(self, 'dataset'):
            self.dataset.close()
        self.dataset = h5py.File(path, 'r')
        self.is_temp = True
        self.init_attrs()

    def filter(self, labels, label_names, example_indices, view_names, path):
        dataset_file_path = os.path.join(path,self.get_name()+"_temp_filter.hdf5")
        new_dataset_file = h5py.File(dataset_file_path,"w")
        self.dataset.copy("Metadata", new_dataset_file)
        new_dataset_file.get("Metadata").attrs["datasetLength"] = len(example_indices)
        new_dataset_file.get("Metadata").attrs["nbClass"] = np.unique(labels)
        new_dataset_file.create_dataset("Labels", data=labels)
        new_dataset_file.get("Labels").attrs["names"] = [label_name.encode()
                                                     if not isinstance(label_name, bytes)
                                                     else label_name
                                                     for label_name in label_names]
        view_names = self.init_view_names(view_names)
        new_dataset_file.get("Metadata").attrs["nbView"] = len(view_names)
        for new_index, view_name in enumerate(view_names):
            self.copy_view(target_dataset=new_dataset_file,
                           source_view_name=view_name,
                           target_view_index=new_index,
                           example_indices=example_indices)
        new_dataset_file.close()
        self.update_hdf5_dataset(dataset_file_path)

    def add_gaussian_noise(self, random_state, path,
                           noise_std=0.15):
        """In this function, we add a guaussian noise centered in 0 with specified
        std to each view, according to it's range (the noise will be
        mutliplied by this range) and we crop the noisy signal according to the
        view's attributes limits.
        This is done by creating a new dataset, to keep clean data."""
        noisy_dataset = h5py.File(path + self.get_name() + "_noised.hdf5", "w")
        self.dataset.copy("Metadata", noisy_dataset)
        self.dataset.copy("Labels", noisy_dataset)
        for view_index in range(self.nb_view):
            self.copy_view(target_dataset=noisy_dataset,
                                   source_view_name=self.get_view_name(view_index),
                                   target_view_index=view_index)
        for view_index in range(noisy_dataset.get("Metadata").attrs["nbView"]):
            view_key = "View" + str(view_index)
            view_dset = noisy_dataset.get(view_key)
            try:
                view_limits = self.dataset[
                    "Metadata/View" + str(view_index) + "_limits"].value
            except:
                import pdb;pdb.set_trace()
            view_ranges = view_limits[:, 1] - view_limits[:, 0]
            normal_dist = random_state.normal(0, noise_std, view_dset.value.shape)
            noise = normal_dist * view_ranges
            noised_data = view_dset.value + noise
            noised_data = np.where(noised_data < view_limits[:, 0],
                                   view_limits[:, 0], noised_data)
            noised_data = np.where(noised_data > view_limits[:, 1],
                                   view_limits[:, 1], noised_data)
            noisy_dataset[view_key][...] = noised_data
        noisy_dataset_path = noisy_dataset.filename
        noisy_dataset.close()
        self.update_hdf5_dataset(noisy_dataset_path)


    # The following methods are hdf5 free

    def select_views_and_labels(self, nb_labels=None,
                                selected_label_names=None, random_state=None,
                                view_names = None, path_for_new="../data/"):
        if view_names is None and selected_label_names is None and nb_labels is None:
            pass
        else:
            selected_label_names = self.check_selected_label_names(nb_labels,
                                                               selected_label_names,
                                                               random_state)
            labels, label_names, example_indices = self.select_labels(selected_label_names)
            self.filter(labels, label_names, example_indices, view_names, path_for_new)
        labels_dictionary = dict(
            (labelIndex, labelName) for labelIndex, labelName in
            enumerate(self.get_label_names()))
        return labels_dictionary

    def get_name(self):
        """Ony works if there are not multiple dots in the files name"""
        return self.dataset.filename.split('/')[-1].split('.')[0]

    def select_labels(self, selected_label_names):
        selected_labels = [self.get_label_names().index(label_name.decode())
                           if isinstance(label_name, bytes)
                           else self.get_label_names().index(label_name)
                                   for label_name in selected_label_names]
        selected_indices = np.array([index
                                     for index, label in enumerate(self.get_labels())
                                     if label in selected_labels])
        labels = np.array([selected_labels.index(self.get_labels()[idx])
                           for idx in selected_indices])
        return labels, selected_label_names, selected_indices

    def check_selected_label_names(self, nb_labels=None,
                                   selected_label_names=None,
                                   random_state=np.random.RandomState(42)):
        if selected_label_names is None or nb_labels is None or len(selected_label_names) < nb_labels:
            if selected_label_names is None:
                nb_labels_to_add = nb_labels
                selected_label_names = []
            elif nb_labels is not None:
                nb_labels_to_add = nb_labels - len(selected_label_names)
            else:
                nb_labels_to_add=0
            labels_names_to_choose = [available_label_name
                                      for available_label_name
                                      in self.get_label_names()
                                      if available_label_name
                                      not in selected_label_names]
            added_labels_names = random_state.choice(labels_names_to_choose,
                                                     nb_labels_to_add,
                                                     replace=False)
            selected_label_names = list(selected_label_names) + list(
                added_labels_names)
        elif len(selected_label_names) > nb_labels:
            selected_label_names = list(
                random_state.choice(selected_label_names, nb_labels,
                                    replace=False))

        return selected_label_names






def datasets_already_exist(pathF, name, nbCores):
    """Used to check if it's necessary to copy datasets"""
    allDatasetExist = True
    for coreIndex in range(nbCores):
        import os.path
        allDatasetExist *= os.path.isfile(
            pathF + name + str(coreIndex) + ".hdf5")
    return allDatasetExist

# def get_v(dataset, view_index, used_indices=None):
# #     """Used to extract a view as a numpy array or a sparse mat from the HDF5 dataset"""
# #     if used_indices is None:
# #         used_indices = range(dataset.get("Metadata").attrs["datasetLength"])
# #     if type(used_indices) is int:
# #         return dataset.get("View" + str(view_index))[used_indices, :]
# #     else:
# #         used_indices = np.array(used_indices)
# #         sorted_indices = np.argsort(used_indices)
# #         used_indices = used_indices[sorted_indices]
# #
# #         if not dataset.get("View" + str(view_index)).attrs["sparse"]:
# #             return dataset.get("View" + str(view_index))[used_indices, :][
# #                    np.argsort(sorted_indices), :]
# #         else:
# #             sparse_mat = sparse.csr_matrix(
# #                 (dataset.get("View" + str(view_index)).get("data").value,
# #                  dataset.get("View" + str(view_index)).get("indices").value,
# #                  dataset.get("View" + str(view_index)).get("indptr").value),
# #                 shape=dataset.get("View" + str(view_index)).attrs["shape"])[
# #                          used_indices, :][
# #                          np.argsort(sorted_indices), :]
# #
# #             return sparse_mat


def get_shape(dataset, view_index):
    """Used to get the dataset shape even if it's sparse"""
    if not dataset.get("View" + str(view_index)).attrs["sparse"]:
        return dataset.get("View" + str(view_index)).shape
    else:
        return dataset.get("View" + str(view_index)).attrs["shape"]


def get_value(dataset):
    """Used to get the value of a view in the HDF5 dataset even if it sparse"""
    if not dataset.attrs["sparse"]:
        return dataset.value
    else:
        sparse_mat = sparse.csr_matrix((dataset.get("data").value,
                                        dataset.get("indices").value,
                                        dataset.get("indptr").value),
                                       shape=dataset.attrs["shape"])
        return sparse_mat


def extract_subset(matrix, used_indices):
    """Used to extract a subset of a matrix even if it's sparse"""
    if sparse.issparse(matrix):
        new_indptr = np.zeros(len(used_indices) + 1, dtype=int)
        oldindptr = matrix.indptr
        for exampleIndexIndex, exampleIndex in enumerate(used_indices):
            new_indptr[exampleIndexIndex + 1] = new_indptr[exampleIndexIndex] + (
                    oldindptr[exampleIndex + 1] - oldindptr[exampleIndex])
        new_data = np.ones(new_indptr[-1], dtype=bool)
        new_indices = np.zeros(new_indptr[-1], dtype=int)
        old_indices = matrix.indices
        for exampleIndexIndex, exampleIndex in enumerate(used_indices):
            new_indices[new_indptr[exampleIndexIndex]:new_indptr[
                exampleIndexIndex + 1]] = old_indices[
                                          oldindptr[exampleIndex]:
                                          oldindptr[exampleIndex + 1]]
        return sparse.csr_matrix((new_data, new_indices, new_indptr),
                                 shape=(len(used_indices), matrix.shape[1]))
    else:
        return matrix[used_indices]


def init_multiple_datasets(path_f, name, nb_cores):
    r"""Used to create copies of the dataset if multicore computation is used.

    This is a temporary solution to fix the sharing memory issue with HDF5 datasets.

    Parameters
    ----------
    path_f : string
        Path to the original dataset directory
    name : string
        Name of the dataset
    nb_cores : int
        The number of threads that the benchmark can use

    Returns
    -------
    datasetFiles : None
        Dictionary resuming which mono- and multiview algorithms which will be used in the benchmark.
    """
    if nb_cores > 1:
        if datasets_already_exist(path_f, name, nb_cores):
            logging.debug(
                "Info:\t Enough copies of the dataset are already available")
            pass
        else:
            logging.debug("Start:\t Creating " + str(
                nb_cores) + " temporary datasets for multiprocessing")
            logging.warning(
                " WARNING : /!\ This may use a lot of HDD storage space : " +
                str(os.path.getsize(path_f + name + ".hdf5") * nb_cores / float(
                    1024) / 1000 / 1000) + " Gbytes /!\ ")
            confirmation = confirm()
            if not confirmation:
                sys.exit(0)
            else:
                dataset_files = copy_hdf5(path_f, name, nb_cores)
                logging.debug("Start:\t Creating datasets for multiprocessing")
                return dataset_files


def copy_hdf5(pathF, name, nbCores):
    """Used to copy a HDF5 database in case of multicore computing"""
    datasetFile = h5py.File(pathF + name + ".hdf5", "r")
    for coreIndex in range(nbCores):
        newDataSet = h5py.File(pathF + name + str(coreIndex) + ".hdf5", "w")
        for dataset in datasetFile:
            datasetFile.copy("/" + dataset, newDataSet["/"])
        newDataSet.close()

def delete_HDF5(benchmarkArgumentsDictionaries, nbCores, dataset):
    """Used to delete temporary copies at the end of the benchmark"""
    if nbCores > 1:
        logging.debug("Start:\t Deleting " + str(
            nbCores) + " temporary datasets for multiprocessing")
        args = benchmarkArgumentsDictionaries[0]["args"]
        logging.debug("Start:\t Deleting datasets for multiprocessing")

        for coreIndex in range(nbCores):
            os.remove(args["Base"]["pathf"] + args["Base"]["name"] + str(coreIndex) + ".hdf5")
    if dataset.is_temp:
        dataset.rm()


def confirm(resp=True, timeout=15):
    """Used to process answer"""
    ans = input_(timeout)
    if not ans:
        return resp
    if ans not in ['y', 'Y', 'n', 'N']:
        print('please enter y or n.')
    if ans == 'y' or ans == 'Y':
        return True
    if ans == 'n' or ans == 'N':
        return False


def input_(timeout=15):
    """used as a UI to stop if too much HDD space will be used"""
    logging.warning("You have " + str(
        timeout) + " seconds to stop the dataset copy by typing n")
    i, o, e = select.select([sys.stdin], [], [], timeout)
    if i:
        return sys.stdin.readline().strip()
    else:
        return "y"

def get_monoview_shared(path, name, view_name, labels_names, classification_indices):
    """ATM is not used with shared memory, but soon :)"""
    hdf5_dataset_file = h5py.File(path + name + ".hdf5", "w")
    X = hdf5_dataset_file.get(view_name).value
    y = hdf5_dataset_file.get("Labels").value
    return X, y
