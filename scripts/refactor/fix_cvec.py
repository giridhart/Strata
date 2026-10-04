import sys
import re

with open("src/kernels/sycl/cvec.dp.cpp", "r") as f:
    text = f.read()

# DPCT leftover replacements
text = text.replace("dpct::permute_sub_group_by_xor", "sycl::permute_group_by_xor")
text = text.replace("dpct::dim3", "sycl::range<3>")

# cur_device just returns 0
text = re.sub(r'int cur_device\(\) \{[\s\S]*?return d;\n\}', 'int cur_device() { return 0; }', text)

# fix malloc_device and memcpy bad translation
bad_alloc_block = """    if (t.dir = sycl::malloc_device<float>(
                             g_dir_host.size(), dpct::get_default_queue()) !=
            0 ||
        t.s = sycl::malloc_device<float>(
                             g_s_host.size(), dpct::get_default_queue()) !=
            0 ||
        t.on = sycl::malloc_device<int>(
                             1, dpct::get_default_queue()) != 0 ||
        dpct::get_default_queue()
                             .memcpy(t.dir, g_dir_host.data(),
                                     g_dir_host.size() * sizeof(float))
                             .wait() != 0 ||
        dpct::get_default_queue()
                .memcpy(t.s, g_s_host.data(), g_s_host.size() * sizeof(float))
                .wait() != 0 ||
        dpct::get_default_queue()
                             .memcpy(t.on, &flag, sizeof(int))
                             .wait() != 0) {"""

good_alloc_block = """
    t.dir = sycl::malloc_device<float>(g_dir_host.size(), dpct::get_default_queue());
    t.s = sycl::malloc_device<float>(g_s_host.size(), dpct::get_default_queue());
    t.on = sycl::malloc_device<int>(1, dpct::get_default_queue());
    
    if (!t.dir || !t.s || !t.on) {
        err = "control vector: device allocation failed";
        return false;
    }
    
    dpct::get_default_queue().memcpy(t.dir, g_dir_host.data(), g_dir_host.size() * sizeof(float)).wait();
    dpct::get_default_queue().memcpy(t.s, g_s_host.data(), g_s_host.size() * sizeof(float)).wait();
    dpct::get_default_queue().memcpy(t.on, &flag, sizeof(int)).wait();
    
    if (false) {"""

text = text.replace(bad_alloc_block, good_alloc_block)

# erase device selection logic inside cvec_set and cvec_free
text = re.sub(r'prev = dpct::get_current_device_id\(\);', '', text)
text = re.sub(r'dpct::select_device\([^\)]*\);', '', text)
text = re.sub(r'dpct::get_current_device\(\)\.queues_wait_and_throw\(\);', 'dpct::get_default_queue().wait();', text)
text = re.sub(r'dpct::dpct_free\(([^,]+), [^\)]+\)', r'sycl::free(\1, dpct::get_default_queue())', text)

with open("src/kernels/sycl/cvec.dp.cpp", "w") as f:
    f.write(text)
with open("src/kernels/sycl/cvec.dp.cpp", "r") as f:
    text = f.read()

text = text.replace("const sycl::range<3> grid((unsigned)g_cvec.hc, (unsigned)T);", "const unsigned blocks = ((unsigned)g_cvec.hc) * ((unsigned)T);")
text = text.replace("sycl::nd_range<1>(grid * sycl::range<1>(THREADS),", "sycl::nd_range<1>(sycl::range<1>(blocks * THREADS),")
text = text.replace("sycl::reqd_sub_group_size(32)", "intel::reqd_sub_group_size(32)")
# Also fix extraneous `)` in `sycl::free`
text = text.replace("sycl::free(g_dev[d].on, ([](){ static sycl::queue q; return q; }())));", "sycl::free(g_dev[d].on, ([](){ static sycl::queue q; return q; }()));")
text = text.replace("sycl::free(g_dev[d].dir, ([](){ static sycl::queue q; return q; }())));", "sycl::free(g_dev[d].dir, ([](){ static sycl::queue q; return q; }()));")
text = text.replace("sycl::free(g_dev[d].s, ([](){ static sycl::queue q; return q; }())));", "sycl::free(g_dev[d].s, ([](){ static sycl::queue q; return q; }()));")
text = text.replace("dpct::get_current_device()", "")

with open("src/kernels/sycl/cvec.dp.cpp", "w") as f:
    f.write(text)
