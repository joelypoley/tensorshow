# Tensorshow

Tensorshow is a python module for inspecting [TFRecords](https://www.tensorflow.org/api_guides/python/reading_data#file_formats).

## Installation

Requires python 3.6.

```bash
pip install tensorshow
```

## Examples

Tensorshow can convert a TFRecord to a pandas dataframe.

```python
import tensorshow

# The column labels of `df` are the features of the tf.train.example protobufs.
df = tensorshow.dataframe_from('path/to/tfrecord')
```
<!---
Tensorshow can be used as a command line utility. It will convert a tfrecord to an html file on the command line.

```bash
python tensorshow --tfrecord=/Users/joel/train.tfrecord --html-file=Users/joel/out.html
```

Images stored as byte strings will be automatically detected and displayed as images rather than text. The `out.html` file looks like this when you open it with a browser.
--->

Tensorshow can convert a tfrecord to an html file.

```python
import tensorshow

# The column labels of `df` are the features of the tf.train.example protobufs.
df = tensorshow.html_file_from('path/to/tfrecord', 'path/to/html/outfile', limit=100)
```

The resulting html file will look like this. Tensorshow automatically detects if a byte string is an encoded image and displays it appropriately.

![TFRecord displayed as a table](http://www.joellaity.com/img/html_tensorshow_example.png)


Tensorshow can be used in a jupyter notebook to preview a TFRecord. The `head` function will show the first five `tf.train.example`s by default and the `sample` function will show five random `tf.train.example`s from the tfrecord.

![A preview of a TFRecord in a jupyter notebook](http://www.joellaity.com/img/nb_tensorshow_example.png)
