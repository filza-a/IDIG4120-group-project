from typing import Self
from pathlib import Path
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from scipy.fft import dctn, idctn
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import json


class Transform:
    name = "transform"

    def __init__(self, block_size=8):
        self.block_size = block_size

    def _padding(self, image):
        height = image.shape[0]
        width = image.shape[1]

        pad_h = (-height) % self.block_size
        pad_w = (-width) % self.block_size

        padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode="edge")

        return padded

    def encode(self, channel):
        blocks = []

        padded = self._padding(channel)

        height = padded.shape[0]
        width = padded.shape[1]

        for row in range(0, height, self.block_size):
            for col in range(0, width, self.block_size):
                block = padded[row : row + self.block_size, col : col + self.block_size]
                blocks.append(block)

        return blocks, {
            "original_shape": channel.shape,
            "padded_h": height,
            "padded_w": width,
            "block_size": self.block_size,
        }

    def decode(self, blocks, metadata):
        blocks = np.asarray(blocks)

        height, width = metadata["original_shape"]
        padded_h = metadata["padded_h"]
        padded_w = metadata["padded_w"]
        block_size = metadata["block_size"]

        padded_image = np.empty((padded_h, padded_w), dtype=blocks.dtype)

        index = 0
        for row in range(0, padded_h, block_size):
            for col in range(0, padded_w, block_size):
                padded_image[row : row + block_size, col : col + block_size] = blocks[
                    index
                ]
                index += 1

        return padded_image[:height, :width]


class Quantisation:
    name = "quantisation"

    def __init__(self, q_steps=8):
        self.q_steps = q_steps

    def encode(self, block):
        # Apply quantisation to the DCT block
        quantised_block = np.round(block / self.q_steps).astype(int)

        return quantised_block, {"q_step": self.q_steps}

    def decode(self, quantised, metadata):
        return np.asarray(quantised, dtype=np.float64) * metadata["q_step"]


class RLE:
    name = "rle"

    def encode(self, values):
        encoded = []
        zero_count = 0

        for value in values:
            value = int(value)

            if value == 0:
                zero_count += 1
            else:
                encoded.append((zero_count, value))
                zero_count = 0

        # Record remaining zeros at the end
        encoded.append((zero_count, None))

        return encoded, {"dtype": values.dtype.str}

    def decode(self, encoded, metadata):
        values = []

        for zero_count, value in encoded:
            values.extend(zero_count * [0])

            if value is None:
                break

            values.append(value)

        return np.array(values, dtype=metadata["dtype"])


class DCT:
    name = "DCT"

    def __init__(self, shift=128):
        self.shift = shift

    def encode(self, block):
        shifted_block = block.astype(np.float64) - self.shift
        dct_block = dctn(shifted_block, norm="ortho")

        return dct_block, {"shift": self.shift}

    def decode(self, coefficients, metadata):
        shifted_block = idctn(coefficients, norm="ortho")
        block = shifted_block + metadata["shift"]

        return block


class ZigZag:
    name = "zigzag"

    def _indices(self, n):
        indices = []

        for diagonal in range(2 * n - 1):
            cells = []

            for row in range(n):
                col = diagonal - row

                if 0 <= col < n:
                    cells.append((row, col))

            if diagonal % 2 == 0:
                cells.reverse()

            indices.extend(cells)

        return indices

    def encode(self, quantised_block):
        block_size = quantised_block.shape[0]
        indices = self._indices(block_size)
        zigzag_values = np.array([quantised_block[row, col] for row, col in indices])

        return zigzag_values, {"block_size": block_size}

    def decode(self, values, metadata):
        n = metadata["block_size"]
        block = np.empty((n, n), dtype=values.dtype)

        for value, (row, col) in zip(values, self._indices(n)):
            block[row, col] = value

        return block


class Codec:
    def __init__(self, stages=None, block_size=8, output_dir="compressed"):
        self.stages = list(stages)
        self.block_size = block_size
        self.block_tranform = Transform(block_size)
        self.output_dir = Path(output_dir)

    def compress(self, image, output_file="compressed_image.json"):

        image_array = np.asarray(image)
        height, width, channels = image_array.shape
        encoded_image = {}
        channel_names = ["Red", "Green", "Blue"]

        metadata = {
            "height": height,
            "width": width,
            "channels": channels,
            "channel_order": channel_names,
            "dtype": "uint8",
        }

        for channel_index, channel_name in enumerate(channel_names):
            channel = image_array[:, :, channel_index]

            blocks, transform_metadata = self.block_tranform.encode(channel)
            metadata["block_size"] = transform_metadata["block_size"]
            metadata["padded_h"] = transform_metadata["padded_h"]
            metadata["padded_w"] = transform_metadata["padded_w"]

            encoded_blocks = []

            for block in blocks:
                stage_records = []
                for stage in self.stages:
                    block, stage_metadata = stage.encode(block)

                    if stage.name == "quantisation":
                        metadata["q_step"] = stage_metadata["q_step"]

                    stage_records.append(
                        {"name": stage.name, "metadata": stage_metadata}
                    )

                encoded_blocks.append(
                    {"payload": block, "block_metadata": stage_records}
                )

            encoded_image[channel_name] = encoded_blocks

        metadata["rle_format"] = "zero_count, value; None marks end of block"

        compressed = {
            "metadata": metadata,
            "encoded_data": encoded_image,
        }

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save inside encoded_outputs folder
        output_file = self.output_dir / Path(output_file).name

        # Save encoded data
        with open(output_file, "w") as f:
            json.dump(compressed, f)

        print(f"Encoded image saved to: {output_file}")

        return compressed

    def decompress(self, compressed):

        encoded_data = compressed["encoded_data"]
        metadata = compressed["metadata"]

        decoded_channels = []

        channel_metadata = {
            "original_shape": (
                metadata["height"],
                metadata["width"],
            ),
            "block_size": metadata["block_size"],
            "padded_h": metadata["padded_h"],
            "padded_w": metadata["padded_w"],
        }

        for channel in metadata["channel_order"]:
            decoded_blocks = []

            encoded_blocks = encoded_data[channel]

            for encoded_block in encoded_blocks:
                block = encoded_block["payload"]
                block_metadata = encoded_block["block_metadata"]

                for stage, stage_data in zip(
                    reversed(self.stages), reversed(block_metadata)
                ):
                    block = stage.decode(block, stage_data["metadata"])

                decoded_blocks.append(block)

            channel = self.block_tranform.decode(decoded_blocks, channel_metadata)

            decoded_channels.append(channel)

        image = np.stack(decoded_channels, axis=-1)

        image = np.clip(np.round(image), 0, 255)

        return image.astype(metadata["dtype"])
