# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import os
from pathlib import Path
from typing import Any, List, Optional
from src.models.data.radym import Radym


def _count_numeric_subdirs(scene_root: Path) -> int:
    """Count immediate subdirs whose names are digits (Lyra view / trajectory indices)."""
    if not scene_root.is_dir():
        return 1
    return max(
        sum(1 for x in scene_root.iterdir() if x.is_dir() and x.name.isdigit()),
        1,
    )


class RadymWrapper(Radym):
    def __init__(self, is_static: bool = True, is_multi_view: bool = False, **kwargs):
        super().__init__(**kwargs)

        # For recon code base
        self.is_static = is_static
        self.is_multi_view = is_multi_view

        # For multi-view datasets, mp4_file_paths may contain one entry per
        # (scene, view) pair (e.g. root/<scene>/<view>/rgb/<scene>.mp4).
        # The multi-view reader only needs one anchor path per scene — it
        # navigates to other views via view_idx.  Keeping duplicates causes
        # the train/test split to group views of the same scene together,
        # producing multiple reconstructions of the same image.
        if is_multi_view and len(self.mp4_file_paths) > 0:
            seen = set()
            unique = []
            for p in self.mp4_file_paths:
                key = p.name          # e.g. "destylized_green_land.mp4"
                if key not in seen:
                    seen.add(key)
                    unique.append(p)
            self.mp4_file_paths = sorted(unique)

        self.sample_list = self.mp4_file_paths
        if is_multi_view:
            # Demo layout: root_path/<view_idx>/rgb/*.mp4  -> scene root = parent of view folder.
            # Nested layout: root_path/<scene>/<view_idx>/rgb/*.mp4 -> scene root still works
            # (parent.parent.parent). Listing root_path alone would count *scenes* as "cameras"
            # and break min(num_input_multi_views, camera_count) vs model output.
            if len(self.mp4_file_paths) > 0:
                scene_root = Path(self.mp4_file_paths[0]).resolve().parent.parent.parent
                self.num_cameras = _count_numeric_subdirs(scene_root)
            else:
                self.num_cameras = len(
                    [
                        camera_name
                        for camera_name in os.listdir(self.root_path)
                        if camera_name != "flag"
                    ]
                )
            self.n_views = self.num_cameras
        else:
            self.num_cameras = 1

    def __len__(self):
        return len(self.sample_list)

    def count_frames(self, video_idx: int):
        return self.num_frames(video_idx)

    def count_cameras(self, video_idx: int):
        if not self.is_multi_view:
            return 1
        if video_idx < 0 or video_idx >= len(self.mp4_file_paths):
            return self.num_cameras
        scene_root = Path(self.mp4_file_paths[video_idx]).resolve().parent.parent.parent
        return _count_numeric_subdirs(scene_root)
    
    def get_data(
        self,
        idx,
        data_fields: List[str],
        frame_indices: Optional[List[int]] = None,
        view_indices: Optional[List[int]] = None,
        camera_convention: str = "opencv",
    ):
        assert camera_convention == 'opencv', f"No support for camera convention {camera_convention}"
        if view_indices is None or len(view_indices) == 0:
            view_indices = list(range(self.count_cameras(idx)))
        final_dict = None
        for view_idx in view_indices:
            output_dict = self._read_data(
                idx, frame_indices, [view_idx], data_fields,
            )
            if final_dict is None:
                final_dict = output_dict
            else:
                for k in final_dict:
                    if k == "__key__":
                        continue
                    final_dict[k] = torch.concatenate([final_dict[k], output_dict[k]])
        return final_dict