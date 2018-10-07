# Tensorshow

Tensorshow is a python library for inspecting [TFRecords](https://www.tensorflow.org/api_guides/python/reading_data#file_formats).

Tensorshow can convert a TFRecord to a pandas dataframe

```python
import tensorshow

# The column labels are the features of the tf.examples.
df = tensorshow.to_dataframe('path/to/tfrecord')
```

Tensorshow can be used as a command line utility. It will convert a tfrecord to an html file on the command line.

```bash
python tensorshow --tfrecord='path/to/tfrecord' --html_file='path/to/html_file'
```

Images stored as byte strings will be automatically detected and displayed as images rather than text. The resulting file looks like this.

[[https://github.com/joelypoley/tensorshow/blob/master/html_table.png|alt=table]]


Tensorshow can be used in a jupyter notebook to preview a tfrecord. The `show_head` function will show the first 5 `tf.train.example`s by default and the `show_random` function will show five random `tf.train.example`s from the tfrecord.

[[https://github.com/joelypoley/tensorshow/blob/master/html_table.png|alt=jupyter-notebook]]





