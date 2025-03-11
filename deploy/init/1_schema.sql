-- 创建数据库 llm_knowledge 并指定编码为 UTF8
CREATE DATABASE llm_knowledge WITH ENCODING 'UTF8' LC_COLLATE='en_US.UTF-8' LC_CTYPE='en_US.UTF-8';

-- 连接至新创建的数据库
\c llm_knowledge

-- 在 llm_knowledge 数据库中创建 schema
CREATE SCHEMA llama_index_vector;
CREATE SCHEMA llama_index_graph;
CREATE SCHEMA langgraph_checkpointer;
CREATE SCHEMA langgraph_store;

-- 添加注释（可选）
COMMENT ON SCHEMA llama_index_vector IS 'Schema for storing vector index data';
COMMENT ON SCHEMA llama_index_graph IS 'Schema for storing graph index data';
COMMENT ON SCHEMA langgraph_checkpointer IS 'Schema for storing checkpoints';
COMMENT ON SCHEMA langgraph_store IS 'Schema for storing language graph data';