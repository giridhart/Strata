import sys

with open("src/kernels/sycl/cvec.dp.cpp", "r") as f:
    text = f.read()

# Restore 3D for cvec_kernel
text = text.replace("sycl::nd_range<1>(grid * sycl::range<1>(THREADS),", "sycl::nd_range<3>(grid * sycl::range<3>(1, 1, THREADS),")
text = text.replace("sycl::range<1>(THREADS)),", "sycl::range<3>(1, 1, THREADS)),")
text = text.replace("sycl::nd_item<1> item", "sycl::nd_item<3> item")

# Fix item dimensions mapping for 3D
# In SYCL 3D, z is 0, y is 1, x is 2. (DPCT mapping)
# Our refactor_ast.py blindly replaced item.get_group(2) with item.get_group(0).
# Let's just fix it for cvec.
text = text.replace("const int c = item.get_group(0);", "const int c = item.get_group(2);")

with open("src/kernels/sycl/cvec.dp.cpp", "w") as f:
    f.write(text)
