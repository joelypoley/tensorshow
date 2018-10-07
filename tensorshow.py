import itertools
import random

import base64
import fleep

from IPython.display import HTML
from io import BytesIO
import pandas as pd
from PIL import Image
import tensorflow as tf

styling = """
<head>
<style>

body {
  font-family: sans-serif;
  font-size: 16px;
  font-weight: 400;
  text-rendering: optimizeLegibility;
}

table {border: none;}
 
th {
  background:#FFFFFF;
  font-size:23px;
  font-weight: 200;
  padding:24px;
  text-align:left;
  vertical-align:middle;
  border:none;
}

  
tr {
  border:none;
  font-size:16px;
  font-weight:bold;
}
 
tr:hover td {
  background:#E0FFFF;
}
 
 
tr:nth-child(odd) td {
  background:#EBEBEB;
}
 
tr:nth-child(odd):hover td {
  background:#E0FFFF;
}

 
td {
  background:#FFFFFF;
  padding:20px;
  text-align:left;
  vertical-align:middle;
  font-weight:300;
  font-size:18px;
  border:none;
}

</style>
</head>
"""


def feature_to_list(feat):
    kind = feat.WhichOneof("kind")
    if kind == "float_list":
        return list(feat.float_list.value)
    elif kind == "int64_list":
        return list(feat.int64_list.value)
    elif kind == "bytes_list":
        return list(feat.bytes_list.value)
    else:
        assert false


def example_to_dict(example_str):
    example = tf.train.Example()
    example.ParseFromString(example_str)
    d = dict(example.features.feature)
    return {k: feature_to_list(v) for k, v in d.items()}


def random_sample(tfrecord_path, max_rows=None):
    if max_rows is None or max_rows < 0:
        return to_dataframe(tfrecord_path, max_rows=max_rows)

    record_it = tf.python_io.tf_record_iterator(tfrecord_path)
    running_sample = list(itertools.islice(record_it, max_rows))
    num_seen_so_far = max_rows
    for x in record_it:
        num_seen_so_far += 1
        idx_to_replace = random.randrange(num_seen_so_far)
        if idx_to_replace < max_rows:
            running_sample[idx_to_replace] = x

    rows = [example_to_dict(record_str) for record_str in running_sample]
    return pd.DataFrame(rows)


def to_dataframe(tfrecord_path, max_rows=None):
    if max_rows is None:
        # 2**63 bytes is ~9 exabytes, so max_rows is essentially infinite.
        max_rows = 1 << 63

    record_it = tf.python_io.tf_record_iterator(tfrecord_path)
    rows = [
        example_to_dict(record_str) for _, record_str in zip(range(max_rows), record_it)
    ]
    return pd.DataFrame(rows)


def image_to_base64(img_str):
    with BytesIO() as img_buffer:
        img_buffer.write(img_str)
        img = Image.open(img_buffer)
        img_buffer.seek(0)
        img.thumbnail((128, 128), Image.ANTIALIAS)
        with BytesIO() as thumb_buffer:
            img.save(thumb_buffer, format="JPEG")
            return str(base64.b64encode(thumb_buffer.getvalue()))[2:-1]


def image_formatter(img):
    return f'<img src="data:image/jpeg;base64,{image_to_base64(img[0])}">'


def cols_with_images(df):
    res = []
    for k in df.keys():
        if isinstance(df[k][0][0], bytes):
            info = fleep.get(df[k][0][0])
            if info.type_matches("raster-image"):
                res.append(k)

    return res


def show_head(tfrecord_path, at_most=5):
    df = to_dataframe(tfrecord_path, max_rows=at_most)
    pd.set_option("display.max_colwidth", -1)
    html_all = df.to_html(
        formatters={col: image_formatter for col in cols_with_images(df)}, escape=False
    )
    return HTML(html_all)


def show_sample(tfrecord_path, at_most=5):
    df = random_sample(tfrecord_path, max_rows=at_most)
    pd.set_option("display.max_colwidth", -1)
    html_all = df.to_html(
        formatters={col: image_formatter for col in cols_with_images(df)}, escape=False
    )
    return HTML(html_all)


def to_html_file(
    tfrecord_path, outfile, at_most=None, random=False, thumbnail_size=None
):
    # TODO: implement thumbnail_size
    if random:
        df = random_sample(tfrecord_path, max_rows=at_most)
    else:
        df = to_dataframe(tfrecord_path, max_rows=at_most)
    pd.set_option("display.max_colwidth", -1)
    with open(outfile, "w+") as f:
        f.write(styling)
        df.to_html(
            buf=f,
            formatters={col: image_formatter for col in cols_with_images(df)},
            escape=False,
        )
