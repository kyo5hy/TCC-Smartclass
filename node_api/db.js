// db.js
const sqlite3 = require('sqlite3').verbose();
const bcrypt = require('bcryptjs');
const path = require('path');
const fs = require('fs');

// Carrega as variáveis do .env localizado na raiz do projeto
require('dotenv').config({ path: path.resolve(__dirname, '../.env') });

const dataPath = path.resolve(__dirname, '../data');
if (!fs.existsSync(dataPath)) fs.mkdirSync(dataPath);

const dbPath = path.join(dataPath, 'database.db');
const db = new sqlite3.Database(dbPath);

db.serialize(() => {
    db.run(`CREATE TABLE IF NOT EXISTS salas (id INTEGER PRIMARY KEY AUTOINCREMENT, nome TEXT UNIQUE, camera_url TEXT DEFAULT '0')`);
    db.run(`CREATE TABLE IF NOT EXISTS zonas_sala (id INTEGER PRIMARY KEY AUTOINCREMENT, sala_id INTEGER, nome_bancada TEXT, coordenadas_json TEXT)`);
    db.run(`CREATE TABLE IF NOT EXISTS turmas (id INTEGER PRIMARY KEY AUTOINCREMENT, nome TEXT, periodo TEXT, ano_semestre TEXT, usuario_id INTEGER DEFAULT 1)`);
    db.run(`CREATE TABLE IF NOT EXISTS materias (id INTEGER PRIMARY KEY AUTOINCREMENT, nome TEXT, turma_id INTEGER, usuario_id INTEGER DEFAULT 1, tempo_evasao_min INTEGER DEFAULT 15)`);
    db.run(`CREATE TABLE IF NOT EXISTS configuracoes_ia (chave TEXT PRIMARY KEY, valor TEXT)`);
    // Migração segura: adiciona a coluna em bancos já existentes (ignora erro se já existir)
    db.run(`ALTER TABLE materias ADD COLUMN tempo_evasao_min INTEGER DEFAULT 15`, () => {});
    db.run(`CREATE TABLE IF NOT EXISTS alunos (matricula TEXT PRIMARY KEY, nome TEXT, turma_id INTEGER, usuario_id INTEGER DEFAULT 1)`);
    db.run(`CREATE TABLE IF NOT EXISTS assentos (id INTEGER PRIMARY KEY AUTOINCREMENT, aluno_matricula TEXT, materia_id INTEGER, sala_id INTEGER, bancada_nome TEXT)`);
    db.run(`CREATE TABLE IF NOT EXISTS aulas (id INTEGER PRIMARY KEY AUTOINCREMENT, materia_id INTEGER, sala_id INTEGER, data_dia TEXT, hora_inicio TEXT, hora_fim TEXT, status TEXT, usuario_id INTEGER DEFAULT 1)`);
    db.run(`CREATE TABLE IF NOT EXISTS presenca (id INTEGER PRIMARY KEY AUTOINCREMENT, aluno_id TEXT, aula_id INTEGER, bancada_atual TEXT, data_hora TEXT, tipo TEXT)`);
    db.run(`CREATE INDEX IF NOT EXISTS idx_presenca_aula_aluno ON presenca(aula_id, aluno_id)`);
    db.run(`CREATE TABLE IF NOT EXISTS alertas (id INTEGER PRIMARY KEY AUTOINCREMENT, aula_id INTEGER, mensagem TEXT, data_hora TEXT, video_file TEXT, image_file TEXT)`);
    db.run(`CREATE INDEX IF NOT EXISTS idx_alertas_aula ON alertas(aula_id)`);
    db.run(`CREATE TABLE IF NOT EXISTS justificativas (id INTEGER PRIMARY KEY AUTOINCREMENT, aula_id INTEGER, aluno_matricula TEXT, motivo TEXT, data_registro TEXT)`);
    db.run(`CREATE TABLE IF NOT EXISTS usuarios (id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT UNIQUE, senha_hash TEXT, nome TEXT, role TEXT)`);
    db.run(`CREATE TABLE IF NOT EXISTS live_status (aluno_matricula TEXT NOT NULL, aula_id INTEGER NOT NULL, status_atencao TEXT, atualizado_em TEXT, bancada_atual TEXT, PRIMARY KEY (aluno_matricula, aula_id))`);
    db.run(`ALTER TABLE live_status ADD COLUMN bancada_atual TEXT`, () => {}); // migração: bancada em tempo real
    // Migração: bancos criados antes da correção multiusuário tinham live_status sem
    // aula_id (dados de "presença ao vivo" de aulas diferentes se misturavam). Esses
    // dados são só um cache temporário (recalculado a cada aula), então é seguro
    // recriar a tabela no formato novo em vez de tentar migrar linha a linha.
    db.get("PRAGMA table_info(live_status)", [], () => {
        db.all("PRAGMA table_info(live_status)", [], (err, cols) => {
            if (!err && cols && !cols.some(c => c.name === 'aula_id')) {
                console.log("🔧 Migrando live_status para o formato multiusuário (aula_id)...");
                db.run("DROP TABLE live_status", () => {
                    db.run(`CREATE TABLE IF NOT EXISTS live_status (aluno_matricula TEXT NOT NULL, aula_id INTEGER NOT NULL, status_atencao TEXT, atualizado_em TEXT, PRIMARY KEY (aluno_matricula, aula_id))`);
                });
            }
        });
    });

    // Verifica e cria o Usuário Master a partir do .env
    db.get("SELECT COUNT(*) as count FROM usuarios", (err, row) => {
        if (row && row.count === 0) {
            const adminEmail = process.env.ADMIN_EMAIL || "admin@admin.com";
            const adminPass = process.env.ADMIN_PASSWORD || "admin123";
            const hash = bcrypt.hashSync(adminPass, 10);
            
            db.run(`INSERT INTO usuarios (email, senha_hash, nome, role) VALUES (?, ?, ?, ?)`, 
                [adminEmail, hash, "Administrador Master", "admin"]);
            console.log(`✅ Usuário Master inicializado: ${adminEmail}`);
        }
    });
});

const runQuery = (sql, params = []) => new Promise((resolve, reject) => db.run(sql, params, function(err) { if(err) reject(err); else resolve(this); }));
const getQuery = (sql, params = []) => new Promise((resolve, reject) => db.get(sql, params, (err, row) => err ? reject(err) : resolve(row)));
const allQuery = (sql, params = []) => new Promise((resolve, reject) => db.all(sql, params, (err, rows) => err ? reject(err) : resolve(rows)));

module.exports = { db, runQuery, getQuery, allQuery };