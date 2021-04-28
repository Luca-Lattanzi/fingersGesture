"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load("//bzl:module.bzl", "isaac_subgraph")
load("//bzl:py.bzl", "isaac_py_app")

isaac_py_app(
    name = "ai_app",
    srcs = ["ai_app.py"],
    data = [
		":ai_app.app.json",
        "//apps:py_init",
        "//apps/tutorials:py_init",
        "//apps/assets/kinematic_trees",
        "//packages/planner/apps:multi_joint_lqr_control_subgraph",
        "//packages/navsim/apps:navsim_tcp_subgraph",
    ],
    modules = [
        "planner",
        "sight",
        "viewers",
    ],
	deps = ["//packages/pyalice"]
)