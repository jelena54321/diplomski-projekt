#include <Python.h>
#include <cstdio>

#define PY_ARRAY_UNIQUE_SYMBOL gen_ARRAY_API
#include "numpy/arrayobject.h"
#include "generate_features.h"

static PyObject* generate_features_cpp(PyObject *self, PyObject *args) {
    char *file_name, *ref, *region;
    if (!PyArg_ParseTuple(args, "sss", &file_name, &ref, &region)) return NULL;

    auto result = generate_features(file_name, ref, region);

    int N = result->positions.size();
    PyObject *positions = PyList_New(N);
    PyObject *X = PyList_New(result->X.size());
    for (int i = 0; i < N; i++) {
        auto& position = result->positions[i];

        int M = position.size();
        PyObject *list = PyList_New(M);
        for (int j = 0; j < M; j++) {
            PyObject *position_tuple = PyTuple_New(2);
            PyTuple_SetItem(position_tuple, 0, PyLong_FromLong(position[j].first));
            PyTuple_SetItem(position_tuple, 1, PyLong_FromLong(position[j].second));

            PyList_SetItem(list, j, position_tuple);
        }

        PyList_SetItem(positions, i, list);
        PyList_SetItem(X, i, result->X[i]);
    }

    PyObject *return_value = PyTuple_New(2);
    PyTuple_SetItem(return_value, 0, positions);
    PyTuple_SetItem(return_value, 1, X);

    return return_value;
}

static PyMethodDef gen_methods[] = {
        {
                "generate_features", generate_features_cpp, METH_VARARGS,
                "Generate features for polisher."
        },
        {NULL, NULL, 0, NULL}
};


static struct PyModuleDef gen_definition = {
        PyModuleDef_HEAD_INIT,
        "gen",
        "Feature generation.",
        -1,
        gen_methods
};


PyMODINIT_FUNC PyInit_gen(void) {
    Py_Initialize();
    import_array();
    return PyModule_Create(&gen_definition);
}