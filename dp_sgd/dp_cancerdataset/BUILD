package(default_visibility = [":internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package_group(
    name = "internal",
    packages = [
        "//differential_privacy/...",
    ],
)

py_binary(
    name = "dp_cancerdataset",
    srcs = [
        "dp_cancerdataset.py",
    ],
    deps = [
        ":adpallocater",
        "//differential_privacy/dp_sgd/dp_optimizer",
        "//differential_privacy/dp_sgd/dp_optimizer:dp_pca",
        "//differential_privacy/dp_sgd/dp_optimizer:utils",
        "//differential_privacy/datasets/cancer:print_csv_tfrecords",

    ],
)


py_library(
    name = "adpallocater",
    srcs = [
        "adpallocater.py",
    ],
    deps = [
    ],
)