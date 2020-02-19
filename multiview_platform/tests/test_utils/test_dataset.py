import unittest
import h5py
import numpy as np
import os

from multiview_platform.tests.utils import rm_tmp, tmp_path
from multiview_platform.mono_multi_view_classifiers.utils import dataset


class Test_Dataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rm_tmp()
        os.mkdir(tmp_path)
        cls.rs = np.random.RandomState(42)
        cls.nb_view = 3
        cls.file_name = "test.hdf5"
        cls.nb_examples = 5
        cls.nb_attr = 7
        cls.nb_class = 3
        cls.views = [cls.rs.randint(0, 10, size=(cls.nb_examples, cls.nb_attr))
                     for _ in range(cls.nb_view)]
        cls.labels = cls.rs.randint(0, cls.nb_class, cls.nb_examples)
        cls.dataset_file = h5py.File(os.path.join(tmp_path, cls.file_name), "w")
        cls.view_names = ["ViewN" + str(index) for index in range(len(cls.views))]
        cls.are_sparse = [False for _ in cls.views]
        for view_index, (view_name, view, is_sparse) in enumerate(
                zip(cls.view_names, cls.views, cls.are_sparse)):
            view_dataset = cls.dataset_file.create_dataset("View" + str(view_index),
                                                           view.shape,
                                                           data=view)
            view_dataset.attrs["name"] = view_name
            view_dataset.attrs["sparse"] = is_sparse
        labels_dataset = cls.dataset_file.create_dataset("Labels",
                                                         shape=cls.labels.shape,
                                                         data=cls.labels)
        cls.labels_names = [str(index) for index in np.unique(cls.labels)]
        labels_dataset.attrs["names"] = [label_name.encode()
                                         for label_name in cls.labels_names]
        meta_data_grp = cls.dataset_file.create_group("Metadata")
        meta_data_grp.attrs["nbView"] = len(cls.views)
        meta_data_grp.attrs["nbClass"] = len(np.unique(cls.labels))
        meta_data_grp.attrs["datasetLength"] = len(cls.labels)

    @classmethod
    def tearDownClass(cls):
        cls.dataset_file.close()

    def test_filter(self):
        """Had to create a new dataset to aviod playing with the class one"""
        file_name = "test_filter.hdf5"
        dataset_file_filter = h5py.File(os.path.join(tmp_path, file_name), "w")
        for view_index, (view_name, view, is_sparse) in enumerate(
                zip(self.view_names, self.views, self.are_sparse)):
            view_dataset = dataset_file_filter.create_dataset(
                "View" + str(view_index),
                view.shape,
                data=view)
            view_dataset.attrs["name"] = view_name
            view_dataset.attrs["sparse"] = is_sparse
        labels_dataset = dataset_file_filter.create_dataset("Labels",
                                                         shape=self.labels.shape,
                                                         data=self.labels)
        labels_dataset.attrs["names"] = [label_name.encode()
                                         for label_name in self.labels_names]
        meta_data_grp = dataset_file_filter.create_group("Metadata")
        meta_data_grp.attrs["nbView"] = len(self.views)
        meta_data_grp.attrs["nbClass"] = len(np.unique(self.labels))
        meta_data_grp.attrs["datasetLength"] = len(self.labels)
        dataset_object = dataset.HDF5Dataset(hdf5_file=dataset_file_filter)
        dataset_object.filter(np.array([0, 1, 0]), ["0", "1"], [1, 2, 3],
                              ["ViewN0"], tmp_path)
        self.assertEqual(dataset_object.nb_view, 1)
        np.testing.assert_array_equal(dataset_object.get_labels(), [0, 1, 0])
        dataset_object.dataset.close()
        os.remove(os.path.join(tmp_path, "test_filter_temp_filter.hdf5"))
        os.remove(os.path.join(tmp_path, "test_filter.hdf5"))

    def test_for_hdf5_file(self):
        dataset_object = dataset.HDF5Dataset(hdf5_file=self.dataset_file)

    def test_from_scratch(self):
        dataset_object = dataset.HDF5Dataset(views=self.views,
                                             labels=self.labels,
                                             are_sparse=self.are_sparse,
                                             file_name="from_scratch"+self.file_name,
                                             view_names=self.view_names,
                                             path=tmp_path,
                                             labels_names=self.labels_names)
        nb_class = dataset_object.get_nb_class()
        self.assertEqual(nb_class, self.nb_class)
        example_indices = dataset_object.init_example_indces()
        self.assertEqual(example_indices, range(self.nb_examples))
        view = dataset_object.get_v(0)
        np.testing.assert_array_equal(view, self.views[0])

    def test_init_example_indices(self):
        example_indices = dataset.HDF5Dataset(hdf5_file=self.dataset_file).init_example_indces()
        self.assertEqual(example_indices, range(self.nb_examples))
        example_indices = dataset.HDF5Dataset(hdf5_file=self.dataset_file).init_example_indces([0,1,2])
        self.assertEqual(example_indices, [0,1,2])

    def test_get_v(self):
        view = dataset.HDF5Dataset(hdf5_file=self.dataset_file).get_v(0)
        np.testing.assert_array_equal(view, self.views[0])
        view = dataset.HDF5Dataset(hdf5_file=self.dataset_file).get_v(1, [0,1,2])
        np.testing.assert_array_equal(view, self.views[1][[0,1,2,], :])

    def test_get_nb_class(self):
        nb_class = dataset.HDF5Dataset(hdf5_file=self.dataset_file).get_nb_class()
        self.assertEqual(nb_class, self.nb_class)
        nb_class = dataset.HDF5Dataset(hdf5_file=self.dataset_file).get_nb_class([0])
        self.assertEqual(nb_class, 1)



    def test_get_view_dict(self):
        dataset_object = dataset.HDF5Dataset(views=self.views,
                                         labels=self.labels,
                                         are_sparse=self.are_sparse,
                                         file_name="from_scratch" + self.file_name,
                                         view_names=self.view_names,
                                         path=tmp_path,
                                         labels_names=self.labels_names)
        self.assertEqual(dataset_object.get_view_dict(), {"ViewN0":0,
                                                          "ViewN1": 1,
                                                          "ViewN2": 2,})

    def test_get_label_names(self):
        dataset_object = dataset.HDF5Dataset(hdf5_file=self.dataset_file)
        raw_label_names = dataset_object.get_label_names(decode=False)
        decoded_label_names = dataset_object.get_label_names()
        restricted_label_names = dataset_object.get_label_names(example_indices=[3,4])
        self.assertEqual(raw_label_names, [b'0', b'1', b'2'])
        self.assertEqual(decoded_label_names, ['0', '1', '2'])
        self.assertEqual(restricted_label_names, ['2'])

    def test_get_nb_exmaples(self):
        dataset_object = dataset.HDF5Dataset(hdf5_file=self.dataset_file)
        nb_examples = dataset_object.get_nb_examples()
        self.assertEqual(nb_examples, self.nb_examples)

    def test_get_labels(self):
        dataset_object = dataset.HDF5Dataset(hdf5_file=self.dataset_file)
        labels = dataset_object.get_labels()
        np.testing.assert_array_equal(labels, self.labels)
        labels = dataset_object.get_labels([1,2,0])
        np.testing.assert_array_equal(labels, self.labels[[1,2,0]])

    def test_copy_view(self):
        dataset_object = dataset.HDF5Dataset(hdf5_file=self.dataset_file)
        new_dataset = h5py.File(os.path.join(tmp_path, "test_copy.hdf5"), "w")
        dataset_object.copy_view(target_dataset=new_dataset,
                                 source_view_name="ViewN0",
                                 target_view_index=1)
        self.assertIn("View1", list(new_dataset.keys()))
        np.testing.assert_array_equal(dataset_object.get_v(0), new_dataset["View1"][()])
        self.assertEqual(new_dataset["View1"].attrs["name"], "ViewN0")
        new_dataset.close()
        os.remove(os.path.join(tmp_path, "test_copy.hdf5"))

    def test_get_name(self):
        dataset_object = dataset.HDF5Dataset(hdf5_file=self.dataset_file)
        self.assertEqual("test", dataset_object.get_name())

    def test_select_labels(self):
        dataset_object = dataset.HDF5Dataset(hdf5_file=self.dataset_file)
        labels, label_names, indices = dataset_object.select_labels(["0", "2"])
        np.testing.assert_array_equal(np.unique(labels), np.array([0,1]))
        self.assertEqual(label_names, ["0","2"])

    def test_check_selected_label_names(self):
        dataset_object = dataset.HDF5Dataset(hdf5_file=self.dataset_file)
        names = dataset_object.check_selected_label_names(nb_labels=2, random_state=self.rs)
        self.assertEqual(names, ["1", "0"])
        names = dataset_object.check_selected_label_names(selected_label_names=['0', '2'],
                                                          random_state=self.rs)
        self.assertEqual(names, ["0", "2"])

    def test_select_views_and_labels(self):
        file_name = "test_filter.hdf5"
        dataset_file_select = h5py.File(os.path.join(tmp_path, file_name), "w")
        for view_index, (view_name, view, is_sparse) in enumerate(
                zip(self.view_names, self.views, self.are_sparse)):
            view_dataset = dataset_file_select.create_dataset(
                "View" + str(view_index),
                view.shape,
                data=view)
            view_dataset.attrs["name"] = view_name
            view_dataset.attrs["sparse"] = is_sparse
        labels_dataset = dataset_file_select.create_dataset("Labels",
                                                            shape=self.labels.shape,
                                                            data=self.labels)
        labels_dataset.attrs["names"] = [label_name.encode()
                                         for label_name in self.labels_names]
        meta_data_grp = dataset_file_select.create_group("Metadata")
        meta_data_grp.attrs["nbView"] = len(self.views)
        meta_data_grp.attrs["nbClass"] = len(np.unique(self.labels))
        meta_data_grp.attrs["datasetLength"] = len(self.labels)
        dataset_object = dataset.HDF5Dataset(hdf5_file=dataset_file_select)
        names = dataset_object.select_views_and_labels(nb_labels=2, view_names=["ViewN0"], random_state=self.rs, path_for_new=tmp_path)
        self.assertEqual(names, {0: '2', 1: '1'})
        self.assertEqual(dataset_object.nb_view, 1)
        dataset_object.dataset.close()
        os.remove(os.path.join(tmp_path, "test_filter_temp_filter.hdf5"))
        os.remove(os.path.join(tmp_path, "test_filter.hdf5"))

    def test_add_gaussian_noise(self):
        file_name = "test_noise.hdf5"
        dataset_file_select = h5py.File(os.path.join(tmp_path, file_name), "w")
        limits = np.zeros((self.nb_attr, 2))
        limits[:, 1] += 100
        meta_data_grp = dataset_file_select.create_group("Metadata")
        for view_index, (view_name, view, is_sparse) in enumerate(
                zip(self.view_names, self.views, self.are_sparse)):
            view_dataset = dataset_file_select.create_dataset(
                "View" + str(view_index),
                view.shape,
                data=view)
            view_dataset.attrs["name"] = view_name
            view_dataset.attrs["sparse"] = is_sparse
            meta_data_grp.create_dataset("View"+str(view_index)+"_limits", data= limits)
        labels_dataset = dataset_file_select.create_dataset("Labels",
                                                            shape=self.labels.shape,
                                                            data=self.labels)
        labels_dataset.attrs["names"] = [label_name.encode()
                                         for label_name in self.labels_names]
        meta_data_grp.attrs["nbView"] = len(self.views)
        meta_data_grp.attrs["nbClass"] = len(np.unique(self.labels))
        meta_data_grp.attrs["datasetLength"] = len(self.labels)
        dataset_object = dataset.HDF5Dataset(hdf5_file=dataset_file_select)
        dataset_object.add_gaussian_noise(self.rs, tmp_path)
        dataset_object.dataset.close()
        os.remove(os.path.join(tmp_path, "test_noise_noised.hdf5"))
        os.remove(os.path.join(tmp_path, "test_noise.hdf5"))


if __name__ == '__main__':
    unittest.main()