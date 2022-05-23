import os.path
import glob

if False:
    vensim_test_files = glob.glob("test-models/tests/*/*.mdl")
    vensim_test_files.sort()

    tests = []
    for file_path in vensim_test_files:
        path, file_name = os.path.split(file_path)
        folder = path.split("/")[-1]

        test_func_string = """
    "%(test_name)s": {
        "folder": "%(folder)s",
        "file": "%(file_name)s"
    },""" % {
            "folder": folder,
            "test_name": folder,
            "file_name": file_name,
        }
        tests.append(test_func_string)

    file_string = """
    vensim_test = {%(tests)s
    }
    """ % {"tests": "".join(tests)}

    with open("test_factory_result.py", "w", encoding="UTF-8") as ofile:
        ofile.write(file_string)

    print("Generated %i integration tests" % len(tests))
