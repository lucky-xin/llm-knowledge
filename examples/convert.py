import os

from langchain_community.document_transformers.openai_functions import create_metadata_tagger


def process_txt_files(src_dir, dst_dir):
    # 确保目标目录存在
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    # 遍历src目录中的所有txt文件
    for filename in os.listdir(src_dir):
        if filename.endswith('.txt'):
            src_path = os.path.join(src_dir, filename)
            dst_path = os.path.join(dst_dir, filename)

            # 读取文件并去除空行
            with open(src_path, 'r', encoding='utf-8') as src_file:
                lines = [line.strip() for line in src_file if line.strip()]

            # 写入处理后的内容到目标文件
            with open(dst_path, 'w', encoding='utf-8') as dst_file:
                dst_file.write('\n'.join(lines))

            print(f"Processed: {filename}")


if __name__ == '__main__':
    src_folder = 'src'  # 源文件夹路径
    dst_folder = 'dst'  # 目标文件夹路径
    process_txt_files(src_folder, dst_folder)
