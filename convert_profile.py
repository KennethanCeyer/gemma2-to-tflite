# Copyright 2024 The AI Edge Torch Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import linecache
import os
import tracemalloc
from pathlib import Path

import ai_edge_torch
import torch
from ai_edge_torch.generative.examples.gemma import gemma2
from ai_edge_torch.generative.quantize import quant_recipes


def display_top(snapshot, key_type='lineno', limit=10):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


def convert_gemma_to_tflite(
    checkpoint_path: str,
    output_path: str,
    prefill_seq_len: int = 512,
    kv_cache_max_len: int = 1024,
    quantize: bool = True,
):
    """Converting a Gemma 2 2B model to multi-signature
    tflite model.
  
    Args:
        checkpoint_path (str): The filepath to the model checkpoint, or directory holding the checkpoint.
        output_path (str): The filepath to the generated tflite model.
        prefill_seq_len (int, optional): The maximum size of prefill input tensor.
          Defaults to 512.
        kv_cache_max_len (int, optional): The maximum size of KV cache buffer,
          including both prefill and decode. Defaults to 1024.
        quantize (bool, optional): Whether the model should be quanized.
          Defaults to True.
    """
    pytorch_model = gemma2.build_2b_model(
        checkpoint_path, kv_cache_max_len=kv_cache_max_len
    )
    # Tensors used to trace the model graph during conversion.
    prefill_tokens = torch.full((1, prefill_seq_len), 0, dtype=torch.long)
    prefill_input_pos = torch.arange(0, prefill_seq_len)
    decode_token = torch.tensor([[0]], dtype=torch.long)
    decode_input_pos = torch.tensor([0], dtype=torch.int64)
  
    # Disabled quantization option for investigating the OOM causes.
    quant_config = None # quant_recipes.full_int8_dynamic_recipe() if quantize else None
    edge_model = (
        ai_edge_torch.signature(
            "prefill", pytorch_model, (prefill_tokens, prefill_input_pos)
        )
        .signature("decode", pytorch_model, (decode_token, decode_input_pos))
        .convert(quant_config=quant_config)
    )
    edge_model.export(output_path)
  
  
if __name__ == "__main__":
    tracemalloc.start()
    project_root = Path(__file__).parent.resolve()
    checkpoint_path = project_root / "model"
    output_path = project_root / "model" / "gemma2.tflite"
    convert_gemma_to_tflite(str(checkpoint_path), str(output_path))
    snapshot = tracemalloc.take_snapshot()
    display_top(snapshot)
  