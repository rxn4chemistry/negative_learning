"""CCC specific code."""

import os
import subprocess


def fix_infiniband():
    ibv = subprocess.run("ibv_devinfo", stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    lines = ibv.stdout.decode("utf-8").split("\n")
    exclude = ""
    for line in lines:
        if "hca_id:" in line:
            name = line.split(":")[1].strip()
        if "\tport:" in line:
            port = line.split(":")[1].strip()
        if "link_layer:" in line and "Ethernet" in line:
            exclude = exclude + f"{name}:{port},"

    if exclude:
        exclude = "^" + exclude[:-1]
        print(exclude)
        os.environ["NCCL_IB_HCA"] = exclude


def set_env():
    # print("Using " + str(torch.cuda.device_count()) + " GPUs---------------------------------------------------------------------")
    LSB_MCPU_HOSTS = os.environ["LSB_MCPU_HOSTS"].split(
        " "
    )  # Parses Node list set by LSF, in format hostname proceeded by number of cores requested
    HOST_LIST = LSB_MCPU_HOSTS[::2]  # Strips the cores per node items in the list
    LSB_JOBID = os.environ[
        "LSB_JOBID"
    ]  # Parses Node list set by LSF, in format hostname proceeded by number of cores requested
    os.environ["MASTER_ADDR"] = HOST_LIST[
        0
    ]  # Sets the MasterNode to thefirst node on the list of hosts
    os.environ["MASTER_PORT"] = "5" + LSB_JOBID[-5:-1]
    os.environ["NODE_RANK"] = str(
        HOST_LIST.index(os.environ["HOSTNAME"])
    )  # Uses the list index for node rank, master node rank must be 0
    os.environ["NCCL_SOCKET_IFNAME"] = "ib,bond"  # avoids using docker of loopback interface
    os.environ["NCCL_DEBUG"] = (
        "INFO"  # sets NCCL debug to info, during distributed training, bugs in code show up as nccl errors
    )
    os.environ["NCCL_IB_CUDA_SUPPORT"] = "1"  # Force use of infiniband
    os.environ["NCCL_TOPO_DUMP_FILE"] = "NCCL_TOP.%h.xml"
    os.environ["NCCL_DEBUG_FILE"] = "NCCL_DEBUG.%h.%p.txt"
