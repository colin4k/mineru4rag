import json
import os
from ragflow_sdk import RAGFlow
from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod

def process_pdf(pdf_file_path):
    """处理PDF文件并返回内容列表"""
    # 参数设置
    name_without_suff = pdf_file_path.split(".")[0]  # 去除文件扩展名
    
    # 准备环境
    local_image_dir, local_md_dir = "output/images", "output"# 图片和输出目录
    image_dir = str(os.path.basename(local_image_dir))  # 获取图片目录名
        
    # 初始化数据写入器
    image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(
        local_md_dir
    )
    
    # 读取PDF文件内容
    reader1 = FileBasedDataReader("")  # 初始化数据读取器
    pdf_bytes = reader1.read(pdf_file_path)  # 读取PDF文件内容为字节流
    
    # 处理流程
    ## 创建PDF数据集实例
    ds = PymuDocDataset(pdf_bytes)  # 使用PDF字节流初始化数据集
    
    ## 推理阶段
    if ds.classify() == SupportedPdfParseMethod.OCR:
        # 如果是OCR类型的PDF（扫描件/图片型PDF）
        infer_result = ds.apply(doc_analyze, ocr=True)  # 应用OCR模式的分析
        ## 处理管道
        pipe_result = infer_result.pipe_ocr_mode(image_writer)  # OCR模式的处理管道
    else:
        # 如果是文本型PDF
        infer_result = ds.apply(doc_analyze, ocr=False)  # 应用普通文本模式的分析
        ## 处理管道
        pipe_result = infer_result.pipe_txt_mode(image_writer)  # 文本模式的处理管道
    
    ### 获取内容列表（JSON格式）
    content_list = pipe_result.get_content_list(image_dir)
    return content_list

def add_chunks_to_ragflow(content_list, api_key, base_url, knowledge_base_name, doc_id):
    """将内容添加到RAGFlow知识库中"""
    rag_object = RAGFlow(api_key=api_key, base_url=base_url)
    dataset = rag_object.list_datasets(name=knowledge_base_name)
    dataset = dataset[0]
    doc = dataset.list_documents(id=doc_id)
    doc = doc[0]
    
    # 遍历内容列表，找出没有text_level的文本内容
    added_count = 0
    for item in content_list:
        if item.get('type') == 'text'and'text_level' not in item:
            content = item.get('text', '')
            if content:
                chunk = doc.add_chunk(content=content)
                print(f"已添加内容: {content[:30]}...")
                added_count += 1
    
    print(f"总共添加了 {added_count} 个文本块")
    return added_count

def main():
    # 配置参数
    pdf_file_name = "有为：汉武帝的五十四年 (戴波).pdf"
    api_key = "ragflow-I4Nzg5ODcwMTVjODExZjBhODI3MDI0Mm"
    base_url = "http://amd:2081"
    knowledge_base_name = "demo"
    doc_id = "5b2a3b3c15c911f0a3180242ac150006"
    
    # 处理PDF并获取内容列表
    content_list_content = process_pdf(pdf_file_name)
    
    # 将内容添加到RAGFlow
    add_chunks_to_ragflow(content_list_content, api_key, base_url, knowledge_base_name, doc_id)

if __name__ == "__main__":
    main()