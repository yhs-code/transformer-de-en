# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import os
from tqdm import tqdm
import xml.etree.ElementTree as ET
from sacremoses import MosesTokenizer
from pathlib import Path
import argparse


def moses_cut(in_file, out_file, lang):
    mt = MosesTokenizer(lang=lang) # 初始化分词器
    out_f = open(out_file, "w", encoding="utf8")
    with open(in_file, "r", encoding="utf8") as f:
        for line in f.readlines():#每读取一行，进行分词，并写入一行到新的文件中
            line = line.strip()
            if not line:
                continue
            cut_line = mt.tokenize(line, return_str=True) # 分词
            out_f.write(cut_line.lower() + "\n") #变为小写，并写入文件
    out_f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser() # 创建解析器
    parser.add_argument(
        "-p",
        "--pair_dir",
        default=None,
        type=str,
        help="The directory which contains language pair files.",
    )
    parser.add_argument(
        "-d",
        "--dest_dir",
        default=None,
        type=str,
        help="The destination directory to save processed train, dev and test file.",
    )
    parser.add_argument("--src_lang", default="de", type=str, help="source language")
    parser.add_argument("--trg_lang", default="en", type=str, help="target language")

    args = parser.parse_args() # 解析参数，args是一个列表，包含了传递的参数值
    if not args.pair_dir:#如果不传参，就抛异常
        raise ValueError("Please specify --pair_dir")
    #判断args.dest_dir是否存在,不存在就创建
    if not os.path.exists(args.dest_dir):
        os.makedirs(args.dest_dir)
    local_data_path = Path(args.pair_dir) # 获取本地数据路径
    data_dir = Path(args.dest_dir) # 获取保存路径
    
    # 分词
    for mode in ["train", "val", "test"]:
        moses_cut(
            local_data_path / f"{mode}.{args.src_lang}", # 读取源语言文件
            data_dir / f"{mode}_src.cut.txt",
            lang=args.src_lang,
        )
        print(f"[{mode}] 源语言文本分词完成")
        moses_cut(
            local_data_path / f"{mode}.{args.trg_lang}", # 读取目标语言文件
            data_dir / f"{mode}_trg.cut.txt",
            lang=args.trg_lang,
        )
        print(f"[{mode}] 目标语言文本分词完成")
    # 创建文件夹，移动读取的文本到刚创建的文件夹里
    # if not data_dir.exists():
    #     data_dir.mkdir(parents=True)
    # for fpath in local_data_path.glob("*.txt"): # 遍历所有分词后的文件,并移动到目标文件夹
    #     fpath.rename(data_dir / fpath.name)
