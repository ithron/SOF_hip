"""SOF_hip dataset."""
import sys
from typing import List

import dataclasses
import tensorflow as tf
import tensorflow_datasets as tfds

_EXCLUDED: List[str] = [
    "SF42054V5H.dcm",
    "SF42058V5H.dcm",
    "SF42113V5H.dcm",
    "SF42102V5H.dcm",
    "SF42055V5H.dcm",
    "SF42114V5H.dcm",
    "SF42016V5H.dcm",
    "SF42060V5H.dcm",
    "SF42046V5H.dcm",
    "SF42062V5H.dcm",
    "SF10779V1H.dcm"
]

_DESCRIPTION = """
The multi-center Study of Osteoporotic Fractures (SOF) has 16 years of prospective data about osteoporosis that has
served as the basis for many findings about osteoporosis and aging in women age 65 or older.
In addition to adjudication of fractures, SOF has tracked cases of incident breast cancer, stroke, and total and
cause-specific mortality. The data include serial measures of bone mineral density, measurements of sex and calcitropic
hormones, tests of strength and function, cognitive exams, use of medication, health habits and much more.

The [SOF Online website](http://sof.ucsf.edu/) was created in 2011 in order to open the SOF data to the broader research community.
In collaboration with SOF Online, the National Sleep Research Resource is making available the SOF raw sleep study (EDF)
data, along with a fuller set of polysomnography variables. Sleep studies were completed on 461 SOF participants at
Visit 8.
"""

_CITATION = """
@article{zhang2018national,
  title={The National Sleep Research Resource: towards a sleep data commons},
  author={Zhang, Guo-Qiang and Cui, Licong and Mueller, Remo and Tao, Shiqiang and Kim, Matthew and Rueschman, Michael and Mariani, Sara and Mobley, Daniel and Redline, Susan},
  journal={Journal of the American Medical Informatics Association},
  volume={25},
  number={10},
  pages={1351--1358},
  year={2018},
  publisher={Oxford University Press}
}
@article{spira2008sleep,
  title={Sleep-disordered breathing and cognition in older women},
  author={Spira, Adam P and Blackwell, Terri and Stone, Katie L and Redline, Susan and Cauley, Jane A and Ancoli-Israel, Sonia and Yaffe, Kristine},
  journal={Journal of the American Geriatrics Society},
  volume={56},
  number={1},
  pages={45--50},
  year={2008},
  publisher={Wiley Online Library}
}
"""


@dataclasses.dataclass
class SOF_hip_config(tfds.core.BuilderConfig):
    num_images: int = sys.maxsize


class SOF_hip(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for SOF_hip dataset."""

    VERSION = tfds.core.Version('1.0.2')
    RELEASE_NOTES = {
        '1.0.2': 'Save example id and visit',
        '1.0.1': 'Fixed black image and import issues',
        '1.0.0': 'Initial release.',
    }

    BUILDER_CONFIGS = [
        SOF_hip_config(name='unsupervised_raw', description='Image data only, unprocessed'),
        SOF_hip_config(name='unsupervised_raw_tiny', description='Image data only, unprocessed, only 1000 examples', num_images=1000)
    ]

    MANUAL_DOWNLOAD_INSTRUCTIONS = """
      Put all dataset directories into `manual_dir/`.
      """

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                'image': tfds.features.Image(shape=(None, None, 1)),
                'id': tfds.features.Tensor(shape=(), dtype=tf.uint64),
                'visit': tfds.features.Tensor(shape=(), dtype=tf.int8)
            }),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=None,  # Set to `None` to disable
            homepage='http://sof.ucsf.edu',
            citation=_CITATION
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""

        path = dl_manager.manual_dir

        return {
            'train': self._generate_examples(path),
        }

    def _generate_examples(self, path):
        """Yields examples."""

        from sof_utils import dicom, misc
        from functools import reduce
        import numpy as np
        from tqdm import tqdm

        num_images = reduce(lambda x, y: x + 1, dicom.list_files(path, recursive=True), 0)
        num_images -= len(_EXCLUDED)

        target_width, target_height = 0, 0
        min_pixel_val, max_pixel_val = sys.maxsize, 0
        for _, width, height, min_val, max_val in tqdm(dicom.list_meta(path, recursive=True),
                                                       desc="Checking files",
                                                       unit=" files",
                                                       total=num_images,
                                                       dynamic_ncols=True):
            target_width = max(target_width, width)
            target_height = max(target_height, height)
            min_pixel_val = min(min_pixel_val, min_val)
            max_pixel_val = max(max_pixel_val, max_val)

        pixel_range = max_pixel_val - min_pixel_val
        num_images -= len(_EXCLUDED)
        assert num_images > 0

        target_num_images = min(self.builder_config.num_images, num_images)
        stride = int(np.floor(num_images / target_num_images))
        max_index = stride * target_num_images
        assert stride >= 1

        included_files = [f for f in map(lambda x: x[1],
                             filter(lambda x: (x[0] % stride == 0) and (x[0] < max_index),
                                    enumerate(filter(lambda p: p.name not in _EXCLUDED,
                                                     dicom.list_files(path, recursive=True)))))]

        for dcm_file in included_files:
            image = (255.0 * (dicom.read_image(str(dcm_file))[..., np.newaxis] - min_pixel_val).astype(
                np.float) /
                     float(pixel_range)).astype(np.uint8)
            image = tf.image.pad_to_bounding_box(image, 0, 0,
                                                 target_width=target_width,
                                                 target_height=target_height)
            image = tf.image.rot90(image).numpy()

            id, visit = misc.id_and_visit_from_filename(dcm_file.name)

            yield dcm_file.name, {
                'image': image,
                'id': id,
                'visit': visit
            }
