import urllib
import zipfile
import os

TESTSUITE_URL = "https://github.com/SDXorg/test-models/archive/master.zip"

try:
    urllib.urlretrieve(TESTSUITE_URL, "testsuite.zip")
    zf = zipfile.ZipFile("testsuite.zip")
    zf.extractall('tests/')

    os.remove("testsuite.zip")

except:
    print 'Failed to get tests'