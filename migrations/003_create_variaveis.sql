-- Migration: Create variaveis table
CREATE TABLE variaveis (
    var_id SERIAL PRIMARY KEY,
    nome VARCHAR(100) NOT NULL
);
