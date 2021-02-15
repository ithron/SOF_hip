"""SOF_hip dataset."""

import sys
from typing import List, Dict

import dataclasses
import tensorflow.compat.v2 as tf
import tensorflow_datasets.public_api as tfds

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

_CLASS_LABELS = [
    "complete",
    "incomplete",
    "implant"
]


@dataclasses.dataclass
class SOF_hip_config(tfds.core.BuilderConfig):
    num_images: int = sys.maxsize
    split_lr: bool = False
    scale: float = 1.0
    type: str = 'unsupervised'
    labeled: bool = True


class SOF_hip(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for SOF_hip dataset."""

    VERSION = tfds.core.Version('1.0.6')
    RELEASE_NOTES = {
        '1.0.6': 'Add unlabeled keypoint detection dataset',
        '1.0.5': 'Enforce dominance of the \'implant\' label over \'incomplete\'; remove \'UpsideDown\' as a label.',
        '1.0.4': 'Add the ability to build different splits',
        '1.0.3': 'Add key-point detection configuration',
        '1.0.2': 'Save example id and visit',
        '1.0.1': 'Fixed black image and import issues',
        '1.0.0': 'Initial release.',
    }

    BUILDER_CONFIGS = [
        SOF_hip_config(name='unsupervised_raw', description='Image data only, unprocessed', labeled=False),
        SOF_hip_config(name='unsupervised_raw_tiny', description='Image data only, unprocessed, only 1000 examples',
                       num_images=1000, labeled=False),
        SOF_hip_config(name='keypoint_detection',
                       description='Images split in left and right half an downscaled by a factor of 10.',
                       split_lr=True, scale=0.1, type='keypoint_detection', labeled=True),
        SOF_hip_config(name='keypoint_detection_unlabeled',
                       description='Images split in left and right half an downscaled by a factor of 10.',
                       split_lr=True, scale=0.1, type='keypoint_detection', labeled=False)
    ]

    MANUAL_DOWNLOAD_INSTRUCTIONS = """
      Put all dataset directories into `manual_dir/`.
      """

    def _info(self) -> tfds.core.DatasetInfo:

        feature_dict = {
            # These are the features of your dataset like images, labels ...
            'image': tfds.features.Image(encoding_format='png', shape=(None, None, 1)),
            'image/filename': tfds.features.Text(),
            'image/id': tf.int64,
            'image/visit': tf.int8
        }

        if self.builder_config.split_lr:
            feature_dict['image/left_right'] = tfds.features.Text()

        if self.builder_config.type == 'keypoint_detection' and self.builder_config.labeled:
            annotation_features = {
                'image/upside_down': tfds.features.Tensor(shape=(), dtype=tf.int64),
                'object/bbox': tfds.features.BBoxFeature(),
                'object/keypoints': tfds.features.Sequence(tfds.features.Tensor(shape=(2,), dtype=tf.float32)),
                'object/class': tfds.features.ClassLabel(names=_CLASS_LABELS)
            }
            feature_dict = {**feature_dict, **annotation_features}

        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(feature_dict),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=None,  # Set to `None` to disable
            homepage='http://sof.ucsf.edu',
            citation=_CITATION
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        import resource
        # Quickfix for an issue where the maximum number of open files is exhausted.
        _, high = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

        path = dl_manager.manual_dir

        if self.builder_config.labeled:
            splits = set([row['split'] for row in self._load_annotations(path)])
        else:
            splits = ['train']

        split_dict = {split: self._generate_examples(path, split) for split in splits}

        return split_dict

    def _generate_examples(self, path, split="train"):
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

        if self.builder_config.type == 'keypoint_detection':

            if self.builder_config.labeled:
                labels = self._load_annotations(path)

                annotation_dict = {}
                for row in labels:
                    key = f"{row['id']}V{row['visit']}{row['left_right']}"
                    if row['split'] == split:
                        annotation_dict[key] = row
                annotated_keys = set(annotation_dict.keys())

                # filter out all unannotated images
                all_inc_files = included_files
                included_files = []
                for f in all_inc_files:
                    sof_id, visit = misc.id_and_visit_from_filename(f.name)
                    lkey = f"{sof_id}V{visit}R"
                    rkey = f"{sof_id}V{visit}L"
                    if lkey in annotated_keys or rkey in annotated_keys:
                        included_files.append(f)

        for dcm_file in included_files:

            sof_id, visit = misc.id_and_visit_from_filename(dcm_file.name)
            image = (255.0 * (dicom.read_image(str(dcm_file))[..., np.newaxis] - min_pixel_val).astype(
                np.float) /
                     float(pixel_range)).astype(np.uint8)
            image = tf.image.pad_to_bounding_box(image, 0, 0,
                                                 target_width=target_width,
                                                 target_height=target_height)
            image = tf.image.rot90(image)

            image = tf.image.resize(image, (tf.math.ceil(image.shape[0] * self.builder_config.scale),
                                            tf.math.ceil(image.shape[1] * self.builder_config.scale)),
                                    antialias=True)

            if self.builder_config.split_lr:
                padding = image.shape[1] % 2
                image = tf.image.pad_to_bounding_box(image, 0, 0, image.shape[0], image.shape[1] + padding)
                image = tf.cast(image, tf.uint8).numpy()

                assert image.shape[1] % 2 == 0

                half_width = int(image.shape[1] / 2)

                left_image = image[:, 0:half_width, ...]
                right_image = tf.image.flip_left_right(image[:, half_width:, ...]).numpy()

                left_key = f"{sof_id}V{visit}R"
                right_key = f"{sof_id}V{visit}L"

                if self.builder_config.labeled:
                    if left_key in annotated_keys:
                        annotation = annotation_dict[left_key]
                        yield left_key, {
                            'image': left_image,
                            'image/filename': dcm_file.name,
                            'image/id': sof_id,
                            'image/visit': visit,
                            'image/left_right': 'R',
                            'image/upside_down': annotation['upside_down'],
                            'object/bbox': self._build_bbox(annotation),
                            'object/keypoints': self._build_keypoints(annotation),
                            'object/class': self._build_class(annotation)
                        }

                    if right_key in annotated_keys:
                        annotation = annotation_dict[right_key]
                        yield right_key, {
                            'image': right_image,
                            'image/filename': dcm_file.name,
                            'image/id': sof_id,
                            'image/visit': visit,
                            'image/left_right': 'L',
                            'image/upside_down': annotation['upside_down'],
                            'object/bbox': self._build_bbox(annotation),
                            'object/keypoints': self._build_keypoints(annotation),
                            'object/class': self._build_class(annotation)
                        }
                else:
                    yield left_key, {
                        'image': left_image,
                        'image/filename': dcm_file.name,
                        'image/id': sof_id,
                        'image/visit': visit,
                        'image/left_right': 'R'
                    }
                    yield right_key, {
                        'image': right_image,
                        'image/filename': dcm_file.name,
                        'image/id': sof_id,
                        'image/visit': visit,
                        'image/left_right': 'L'
                    }

            else:
                image = tf.cast(image, tf.uint8).numpy()

                key = f"{sof_id}V{visit}"

                yield key, {
                    'image': image,
                    'image/filename': dcm_file.name,
                    'image/id': sof_id,
                    'image/visit': visit
                }

    def _load_annotations(self, path):
        from csv import DictReader
        from pathlib import Path
        # Load label CSV
        with open(Path(path).joinpath('SOF-keypoint-detection-labels.csv'), 'r') as f:
            reader = DictReader(f)
            labels = [self.format_row(row) for row in reader]
        return labels

    @staticmethod
    def format_row(row: Dict) -> Dict:
        new_row = {
            'id': int(row['id']),
            'visit': int(row['visit']),
            'left_right': row['left_right'],
            'filename': row['filename'],
            'upside_down': int(row['upside_down']),
            'incomplete': int(row['incomplete']),
            'width': int(row['width']),
            'height': int(row['height']),
            'implant': int(row['implant']),
            'bbox_min_x': float(row['bbox_min_x']) / 100,
            'bbox_max_x': float(row['bbox_max_x']) / 100,
            'bbox_min_y': float(row['bbox_min_y']) / 100,
            'bbox_max_y': float(row['bbox_max_y']) / 100,
        }
        if "split" in row:
            new_row["split"] = row["split"]
        else:
            new_row["split"] = "train"

        for i in range(12):
            new_row[f"keypoint_x_{i}"] = float(row[f"keypoint_x_{i}"]) / 100
            new_row[f"keypoint_y_{i}"] = float(row[f"keypoint_y_{i}"]) / 100

        return new_row

    @staticmethod
    def _build_bbox(annotation: Dict) -> tfds.features.BBox:
        return tfds.features.BBox(
            xmin=annotation['bbox_min_x'],
            xmax=annotation['bbox_max_x'],
            ymin=annotation['bbox_min_y'],
            ymax=annotation['bbox_max_y'])

    @staticmethod
    def _build_keypoints(annotation: Dict) -> List[List[float]]:
        return [
            [annotation[f"keypoint_y_{i}"], annotation[f"keypoint_x_{i}"]]
            for i in range(12)
        ]

    @staticmethod
    def _build_class(annotation: Dict) -> int:
        label = int(0)
        if annotation['incomplete'] > 0:
            label = int(1)
        if annotation['implant'] > 0:
            label = int(2)
        return label
