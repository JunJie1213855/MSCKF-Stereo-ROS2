#!/usr/bin/env bash
# Source after ROS/workspace setup. All local camera, VIO, bag and RViz
# processes must use the same ROS_DOMAIN_ID and this DDS configuration.
_stereo_recon_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$_stereo_recon_script_dir/setup_tensorrt_env.sh" || return 1
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file://$_stereo_recon_script_dir/../config/cyclonedds_local.xml"
# The XML restricts transport to loopback. Leave the RMW localhost override
# disabled so it cannot replace the explicit interface/peer configuration.
export ROS_LOCALHOST_ONLY=0
unset _stereo_recon_script_dir
