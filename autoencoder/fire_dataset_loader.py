import torch
import torchvision.transforms as T
from tfrecord.torch.dataset import MultiTFRecordDataset
import re


class FireDatasetLoader:
    """The notebook loads the Next Day Wildfire Spread dataset provided on
    kaggle https://www.kaggle.com/datasets/fantineh/next-day-wildfire-spread.
    I used some code snippts from the following kaggle notebook
    https://www.kaggle.com/code/fantineh/data-reader-and-visualization.
    """

    # Data statistics
    # For each variable, the statistics are ordered in the form:
    # (min_clip, max_clip, mean, standard deviation)
    _data_stats = {
        # Elevation in m.
        # 0.1 percentile, 99.9 percentile
        "elevation": (0.0, 3141.0, 657.3003, 649.0147),
        # Pressure
        # 0.1 percentile, 99.9 percentile
        "pdsi": (-6.12974870967865, 7.876040384292651, -0.0052714925, 2.6823447),
        "NDVI": (-9821.0, 9996.0, 5157.625, 2466.6677),  # min, max
        # Precipitation in mm.
        # Negative values do not make sense, so min is set to 0.
        # 0., 99.9 percentile
        "pr": (0.0, 44.53038024902344, 1.7398051, 4.482833),
        # Specific humidity.
        # Negative values do not make sense, so min is set to 0.
        # The range of specific humidity is up to 100% so max is 1.
        "sph": (0.0, 1.0, 0.0071658953, 0.0042835088),
        # Wind direction in degrees clockwise from north.
        # Thus min set to 0 and max set to 360.
        "th": (0.0, 360.0, 190.32976, 72.59854),
        # Min/max temperature in Kelvin.
        # -20 degree C, 99.9 percentile
        "tmmn": (253.15, 298.94891357421875, 281.08768, 8.982386),
        # -20 degree C, 99.9 percentile
        "tmmx": (253.15, 315.09228515625, 295.17383, 9.815496),
        # Wind speed in m/s.
        # Negative values do not make sense, given there is a wind direction.
        # 0., 99.9 percentile
        "vs": (0.0, 10.024310074806237, 3.8500874, 1.4109988),
        # NFDRS fire danger index energy release component expressed in BTU's per
        # square foot.
        # Negative values do not make sense. Thus min set to zero.
        # 0., 99.9 percentile
        "erc": (0.0, 106.24891662597656, 37.326267, 20.846027),
        # Population density
        # min, 99.9 percentile
        "population": (0.0, 2534.06298828125, 25.531384, 154.72331),
        # We don't want to normalize the FireMasks.
        # 1 indicates fire, 0 no fire, -1 unlabeled data
        "PrevFireMask": (-1.0, 1.0, 0.0, 1.0),
        "FireMask": (-1.0, 1.0, 0.0, 1.0),
    }

    def __init__(
        self,
        data_pattern,
        index_pattern,
        splits,
        input_features,
        output_features,
        description=None,
        transform=None,
        batch_size=100,
        data_size=64,
        sample_size=32,
        num_in_channels=11,
        num_out_channels=1,
        clip_and_normalize=False,
        clip_and_rescale=False,
        random_crop=False,
        center_crop=False,
    ):
        """
        Args:
            data_pattern: unix glob pattern for dataset tfrecord files location.
                The pattern is a string with a placeholder indicated by {}.
                Ex. "/tmp/next_day_wildfire_spread_train_{}.tfrecord"
            index_pattern: unix glob pattern for index files location.
                Index file must be provided when using multiple workers, otherwise
                the loader may return duplicate records.
                The pattern is a string with a placeholder indicated by {}.
                Ex. "/tmp/next_day_wildfire_spread_train_{}.index"
            splits: a dictionary of (key, value) pairs, where the key is used to
                construct the data and index path(s) and the value determines
                the contribution of each split to the batch.
                Ex. splits = {
                        "dataset1": 0.8,
                        "dataset2": 0.2,
                    }
            description: a dict of (key, value) pairs to extract from each
                record. The keys represent the name of the features and the
                values ("byte", "float", or "int") correspond to the data type.
                If None (default),then all features contained in the file are extracted.
            input_features: list of input feature names
            output_features: list of output features names
            transform: call back that accepts sample feature map dictionary.
            batch_size: size of each batch.
            sample_size: side length (square) to crop to.
            data_size: side length (sqaure) of input image.
            num_in_channels: number of channels in input_img.
            num_out_channels: number of channels in output_img.
            clip_and_normalize: bool indicating if data should be clipped and normalized
            clip_and_rescale: bool indicating if data should be clipped and rescaled
            random_crop: bool indicating if data should be random cropped
            center_crop: bool indicating if data should be center crop
        Returns:
            None
        """
        self.data_pattern = data_pattern
        self.index_pattern = index_pattern
        self.splits = splits
        self.input_features = input_features
        self.output_features = output_features
        self.description = description
        self.transform = self._parse_sample if transform is None else transform
        self.batch_size = batch_size
        self.data_size = data_size
        self.sample_size = sample_size
        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.clip_and_normalize = clip_and_normalize
        self.clip_and_rescale = clip_and_rescale
        self.random_crop = random_crop
        self.center_crop = center_crop

    def get_loader(self):
        dataset = MultiTFRecordDataset(
            data_pattern=self.data_pattern,
            index_pattern=self.index_pattern,
            splits=self.splits,
            description=self.description,
            transform=self.transform,
            infinite=False,
        )
        return torch.utils.data.DataLoader(dataset, self.batch_size)

    def _random_crop_input_and_output_images(
        self,
        input_img: torch.Tensor,
        output_img: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Randomly axis-align crop input and output image tensors.

        Args:
        input_img: tensor with dimensions HWC.
        output_img: tensor with dimensions HWC.
        Returns:
        input_img: tensor with dimensions HWC.
        output_img: tensor with dimensions HWC.
        """
        combined = torch.concat([input_img, output_img], axis=2)
        combined = torch.permute(combined, (2, 1, 0))
        transform = T.RandomCrop((self.sample_size, self.sample_size))
        combined = transform(combined)
        combined = torch.permute(combined, (2, 1, 0))
        input_img = combined[:, :, 0 : self.num_in_channels + 1]
        output_img = combined[:, :, -self.num_out_channels :]
        return input_img, output_img

    def _center_crop_input_and_output_images(
        self,
        input_img: torch.Tensor,
        output_img: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Center crops input and output image tensors.

        Args:
            input_img: tensor with dimensions HWC.
            output_img: tensor with dimensions HWC.
        Returns:
            input_img: tensor with dimensions HWC.
            output_img: tensor with dimensions HWC.
        """
        transform = T.CenterCrop((self.sample_size, self.sample_size))
        input_img = torch.permute(
            transform(torch.permute(input_img, [2, 1, 0])), [2, 1, 0]
        )
        output_img = torch.permute(
            transform(torch.permute(output_img, [2, 1, 0])), [2, 1, 0]
        )
        return input_img, output_img

    def _get_base_key(self, key: str) -> str:
        """Extracts the base key from the provided key.

        Earth Engine exports TFRecords containing each data variable with its
        corresponding variable name. In the case of time sequences, the name of the
        data variable is of the form 'variable_1', 'variable_2', ..., 'variable_n',
        where 'variable' is the name of the variable, and n the number of elements
        in the time sequence. Extracting the base key ensures that each step of the
        time sequence goes through the same normalization steps.
        The base key obeys the following naming pattern: '([a-zA-Z]+)'
        For instance, for an input key 'variable_1', this function returns 'variable'.
        For an input key 'variable', this function simply returns 'variable'.

        Args:
            key: Input key.

        Returns:
            The corresponding base key.

        Raises:
            ValueError when `key` does not match the expected pattern.
        """
        match = re.match(r"([a-zA-Z]+)", key)
        if match:
            return match.group(1)
        raise ValueError(f"The provided key does not match the expected pattern: {key}")

    def _clip_and_rescale(self, inputs: torch.Tensor, key: str) -> torch.Tensor:
        """Clips and rescales inputs with the stats corresponding to `key`.

        Args:
            inputs: Inputs to clip and rescale.
            key: Key describing the inputs.

        Returns:
            Clipped and rescaled input.

        Raises:
            ValueError if there are no data statistics available for `key`.
        """
        base_key = self._get_base_key(key)
        if base_key not in self._data_stats:
            raise ValueError(
                f"No data statistics available for the requested key: {key}."
            )
        min_val, max_val, _, _ = self._data_stats[base_key]
        inputs = torch.clamp(inputs, min_val, max_val)
        return torch.nan_to_num(
            torch.div((inputs - min_val), (max_val - min_val)), 0, 0, 0
        )

    def _clip_and_normalize(self, inputs: torch.Tensor, key: str) -> torch.Tensor:
        """Clips and normalizes inputs with the stats corresponding to `key`.

        Args:
            inputs: Inputs to clip and normalize.
            key: Key describing the inputs.

        Returns:
            Clipped and normalized input.

        Raises:
            ValueError if there are no data statistics available for `key`.
        """
        base_key = self._get_base_key(key)
        if base_key not in self._data_stats:
            raise ValueError(
                f"No data statistics available for the requested key: {key}."
            )
        min_val, max_val, mean, std = self._data_stats[base_key]
        inputs = torch.clamp(inputs, min_val, max_val)
        inputs = inputs - mean
        return torch.nan_to_num(torch.div(inputs, std))

    def _parse_sample(
        self, samples: dict[str, torch.tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reads a serialized example.

        Args:
        samples: A dict with TensorFlow example protobuf data.

        Returns:
        (input_img, output_img) tuple of inputs and outputs to the ML model.
        """
        if self.random_crop and self.center_crop:
            raise ValueError("Cannot have both random_crop and center_crop be True")

        # convert numpy array to a Pytorch Tensor or shape (data_size, data_size)
        features = {
            k: torch.reshape(torch.Tensor(x), (self.data_size, self.data_size))
            for k, x in samples.items()
        }

        if self.clip_and_normalize:
            inputs_list = [
                self._clip_and_normalize(features.get(key), key)
                for key in self.input_features
            ]
        elif self.clip_and_rescale:
            inputs_list = [
                self._clip_and_rescale(features.get(key), key)
                for key in self.input_features
            ]
        else:
            inputs_list = [features.get(key) for key in self.input_features]

        inputs_stacked = torch.stack(inputs_list, axis=0)
        input_img = torch.permute(inputs_stacked, (1, 2, 0))

        outputs_list = [features.get(key) for key in self.output_features]
        assert outputs_list, "outputs_list should not be empty"
        outputs_stacked = torch.stack(outputs_list, axis=0)

        outputs_stacked_shape = list(outputs_stacked.shape)
        assert len(outputs_stacked.shape) == 3, (
            "outputs_stacked should be rank 3"
            "but dimensions of outputs_stacked"
            f" are {outputs_stacked_shape}"
        )
        output_img = torch.permute(outputs_stacked, (1, 2, 0))

        if self.random_crop:
            input_img, output_img = self._random_crop_input_and_output_images(
                input_img, output_img
            )
        if self.center_crop:
            input_img, output_img = self._center_crop_input_and_output_images(
                input_img, output_img
            )
        return input_img, output_img
