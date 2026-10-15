// server.js
const express = require('express');
const cors = require('cors');
const jwt = require('jsonwebtoken');
const bcrypt = require('bcryptjs');
const fs = require('fs');
const path = require('path');
const axios = require('axios'); 
const ExcelJS = require('exceljs');

// Carrega as variáveis do .env localizado na raiz do projeto
require('dotenv').config({ path: path.resolve(__dirname, '../.env') });

const { runQuery, getQuery, allQuery } = require('./db');

// URL do serviço de IA (Python/FastAPI).
// Em Docker Compose, os containers se comunicam pelo nome do serviço na rede
// interna (ex: "python_ia"), não por 127.0.0.1. Fora do Docker (execução local
// tradicional, como antes), continua caindo no 127.0.0.1:8000 de sempre.
const PYTHON_IA_URL = process.env.PYTHON_IA_URL || 'http://127.0.0.1:8000';

const app = express();
app.use(cors());
// Cabeçalhos básicos de segurança HTTP — corrige a ausência total encontrada no
// teste de segurança (sem isso, o sistema podia ser embutido em um iframe
// malicioso em outro site — clickjacking — entre outras exposições comuns).
app.use((req, res, next) => {
    res.setHeader('X-Frame-Options', 'DENY');
    res.setHeader('X-Content-Type-Options', 'nosniff');
    res.setHeader('Referrer-Policy', 'same-origin');
    next();
});
app.use(express.json({ limit: '10mb' })); 

// --- SERVIDOR DO FRONTEND ---
// Ajustado para apontar para a pasta correta do seu projeto: 'frontend'
const frontendPath = path.resolve(__dirname, '../frontend');
if (!fs.existsSync(frontendPath)) fs.mkdirSync(frontendPath, { recursive: true });
app.use(express.static(frontendPath));
// ---------------------------

// Criando e Liberando diretório Público para os Clipes e Screenshots
const clipsPath = path.resolve(__dirname, '../data/clips');
if (!fs.existsSync(clipsPath)) fs.mkdirSync(clipsPath, { recursive: true });
app.use('/clips', express.static(clipsPath));

const screenshotsPath = path.resolve(__dirname, '../data/screenshots');
if (!fs.existsSync(screenshotsPath)) fs.mkdirSync(screenshotsPath, { recursive: true });
app.use('/screenshots', express.static(screenshotsPath));

// Consumindo a chave do ambiente de forma segura
const SECRET_KEY = process.env.SECRET_KEY;
if (!SECRET_KEY) {
    console.error("❌ ERRO FATAL: SECRET_KEY não definida no arquivo .env");
    process.exit(1);
}

const authMiddleware = async (req, res, next) => {
    try {
        const token = req.headers.authorization?.split(' ')[1];
        if (!token) return res.status(401).json({ error: "Token não fornecido" });
        const decoded = jwt.verify(token, SECRET_KEY);
        const user = await getQuery("SELECT * FROM usuarios WHERE id = ?", [decoded.id]);
        if (!user) return res.status(401).json({ error: "Usuário inválido" });
        req.user = user;
        next();
    } catch (e) { res.status(401).json({ error: "Sessão inválida" }); }
};

// Corrige Path Traversal: matrícula vira nome de arquivo em data/faces/. Sem essa
// validação, um valor como "../../../../home/pi/x" gravaria arquivo fora da pasta.
// Só permite letras, números, hífen e underscore — formato normal de matrícula.
function matriculaSegura(m) {
    const limpa = String(m || "").trim();
    return /^[a-zA-Z0-9_-]{1,50}$/.test(limpa) ? limpa : null;
}

const adminMiddleware = (req, res, next) => {
    if (req.user.role !== 'admin') return res.status(403).json({ error: "Acesso negado" });
    next();
};

// Corrige Brute Force: sem isso, um atacante na rede local podia tentar senhas
// ilimitadamente contra /token. Bloqueia por IP+email após 5 tentativas erradas
// em 10 minutos, por 10 minutos. Simples e sem dependência externa nova.
const tentativasLogin = new Map(); // chave: "ip|email" -> { erros: n, bloqueadoAte: timestamp|null, janelaInicio: timestamp }
const LOGIN_MAX_TENTATIVAS = 5;
const LOGIN_JANELA_MS = 10 * 60 * 1000;
const LOGIN_BLOQUEIO_MS = 10 * 60 * 1000;

app.post("/token", async (req, res) => {
    const { username, password } = req.body;
    if (!username || !password) return res.status(400).json({ error: "Informe usuário e senha." });
    const chave = `${req.ip}|${username.trim().toLowerCase()}`;
    const agora = Date.now();
    const registro = tentativasLogin.get(chave);
    if (registro && registro.bloqueadoAte && agora < registro.bloqueadoAte) {
        const restamMin = Math.ceil((registro.bloqueadoAte - agora) / 60000);
        return res.status(429).json({ error: `Muitas tentativas erradas. Tente novamente em ${restamMin} minuto(s).` });
    }
    const user = await getQuery("SELECT * FROM usuarios WHERE email = ?", [username.trim()]);
    if (!user || !bcrypt.compareSync(password, user.senha_hash)) {
        const r = (registro && agora - registro.janelaInicio < LOGIN_JANELA_MS) ? registro : { erros: 0, janelaInicio: agora, bloqueadoAte: null };
        r.erros += 1;
        if (r.erros >= LOGIN_MAX_TENTATIVAS) r.bloqueadoAte = agora + LOGIN_BLOQUEIO_MS;
        tentativasLogin.set(chave, r);
        return res.status(401).json({ error: "Credenciais inválidas" });
    }
    tentativasLogin.delete(chave); // login certo: zera o contador de tentativas
    const token = jwt.sign({ id: user.id, email: user.email }, SECRET_KEY, { expiresIn: '24h' });
    res.json({ access_token: token, role: user.role, nome: user.nome });
});

app.get("/usuarios", authMiddleware, adminMiddleware, async (req, res) => res.json(await allQuery("SELECT id, email, nome, role FROM usuarios")));
app.post("/usuarios", authMiddleware, adminMiddleware, async (req, res) => {
    try { await runQuery("INSERT INTO usuarios (email, senha_hash, nome, role) VALUES (?, ?, ?, ?)", [req.body.email.trim(), bcrypt.hashSync(req.body.senha, 10), req.body.nome, req.body.role]); res.json({ status: "ok" }); } 
    catch (e) { res.status(400).json({ error: "Email já existe" }); }
});
app.delete("/usuarios/:id", authMiddleware, adminMiddleware, async (req, res) => {
    if (req.user.id == req.params.id) return res.status(400).json({ error: "Não pode se excluir" });
    await runQuery("DELETE FROM usuarios WHERE id = ?", [req.params.id]); res.json({ status: "ok" });
});

app.get("/salas", authMiddleware, async (req, res) => {
    const salas = await allQuery("SELECT * FROM salas");
    // Falha de segurança corrigida: o link RTSP de uma câmera normalmente vem com
    // usuário/senha embutidos (rtsp://admin:senha@ip:porta/...). Antes, QUALQUER
    // usuário logado (inclusive professores) recebia esse campo do backend, mesmo
    // que a tela só escondesse o menu — bastava chamar a API diretamente para ver
    // a senha da câmera. Agora só o Administrador Master recebe o camera_url; para
    // os demais perfis o campo some por completo (o professor só precisa do nome/id
    // da sala para escolher onde dar aula, nunca da credencial da câmera).
    if (req.user.role !== 'admin') {
        salas.forEach(s => { delete s.camera_url; });
    }
    res.json(salas);
});
// Corrige SSRF: camera_url é buscada pelo servidor via OpenCV e transmitida de
// volta ao navegador. Antes, QUALQUER professor logado podia cadastrar uma "sala"
// apontando para um endereço interno da rede (ex: painel do roteador) e usar o
// próprio SmartClass como proxy para acessá-lo. Agora só o Master configura salas.
app.post("/salas", authMiddleware, adminMiddleware, async (req, res) => { await runQuery("INSERT INTO salas (nome, camera_url) VALUES (?, ?)", [req.body.nome.trim(), req.body.camera_url.trim()]); res.json({ status: "ok" }); });
app.delete("/salas/:id", authMiddleware, adminMiddleware, async (req, res) => {
    // Salas são recurso compartilhado entre todos os professores. Sem essa checagem,
    // qualquer um podia apagar uma sala que outro professor está usando AGORA e
    // derrubar a aula dele no meio.
    const emUso = await getQuery(`SELECT id FROM aulas WHERE sala_id = ? AND status = "EM_ANDAMENTO"`, [req.params.id]);
    if (emUso) return res.status(409).json({ error: "Esta sala está em uso em uma aula agora e não pode ser removida." });
    await runQuery("DELETE FROM zonas_sala WHERE sala_id=?", [req.params.id]);
    await runQuery("DELETE FROM salas WHERE id=?", [req.params.id]);
    res.json({ status: "ok" });
});
app.get("/salas/:id/zonas", authMiddleware, async (req, res) => {
    const zonas = await allQuery("SELECT nome_bancada, coordenadas_json FROM zonas_sala WHERE sala_id = ?", [req.params.id]);
    res.json(zonas.map(z => ({ nome_bancada: z.nome_bancada, coordenadas: JSON.parse(z.coordenadas_json) })));
});
app.post("/salas/:id/zonas", authMiddleware, async (req, res) => {
    await runQuery("DELETE FROM zonas_sala WHERE sala_id = ?", [req.params.id]);
    for (let z of req.body.zonas) await runQuery("INSERT INTO zonas_sala (sala_id, nome_bancada, coordenadas_json) VALUES (?, ?, ?)", [req.params.id, z.nome_bancada, JSON.stringify(z.coordenadas)]);
    res.json({ status: "ok" });
});

app.get("/turmas", authMiddleware, async (req, res) => res.json(await allQuery("SELECT * FROM turmas WHERE usuario_id = ?", [req.user.id])));
app.post("/turmas", authMiddleware, async (req, res) => { await runQuery("INSERT INTO turmas (nome, periodo, ano_semestre, usuario_id) VALUES (?, ?, ?, ?)", [req.body.nome.trim(), req.body.periodo, req.body.ano_semestre, req.user.id]); res.json({ status: "ok" }); });
app.delete("/turmas/:id", authMiddleware, async (req, res) => { await runQuery("DELETE FROM turmas WHERE id=? AND usuario_id=?", [req.params.id, req.user.id]); res.json({ status: "ok" }); });

app.get("/materias", authMiddleware, async (req, res) => res.json(await allQuery("SELECT m.id, m.nome, m.tempo_evasao_min, t.nome as turma_nome FROM materias m JOIN turmas t ON m.turma_id = t.id WHERE m.usuario_id = ?", [req.user.id])));
app.post("/materias", authMiddleware, async (req, res) => { await runQuery("INSERT INTO materias (nome, turma_id, usuario_id) VALUES (?, ?, ?)", [req.body.nome.trim(), req.body.turma_id, req.user.id]); res.json({ status: "ok" }); });
app.delete("/materias/:id", authMiddleware, async (req, res) => { await runQuery("DELETE FROM materias WHERE id=? AND usuario_id=?", [req.params.id, req.user.id]); res.json({ status: "ok" }); });

// Tempo de evasão (minutos) — cada professor define por matéria/turma.
app.put("/materias/:id/evasao", authMiddleware, async (req, res) => {
    const minutos = parseInt(req.body.tempo_evasao_min, 10);
    if (!minutos || minutos < 1 || minutos > 180) return res.status(400).json({ error: "Informe um valor entre 1 e 180 minutos." });
    await runQuery("UPDATE materias SET tempo_evasao_min = ? WHERE id = ? AND usuario_id = ?", [minutos, req.params.id, req.user.id]);
    res.json({ status: "ok" });
});

// Configurações de sensibilidade da IA (celular, ação suspeita, facial) — apenas Administrador Master.
const CONFIG_IA_DEFAULTS = {
    face_conf: "0.60",       // confiança mínima para detectar um rosto (0.1 a 0.9) — 0.6 é o ponto de equilíbrio recomendado
    phone_conf: "0.60",      // confiança mínima para detectar um celular (0.1 a 0.9) — 0.6 é o ponto de equilíbrio recomendado (menor detecta mais fácil, porém mais falso positivo)
    tolerancia_facial: "0.48", // distância máxima para aceitar um rosto como reconhecido (0.30 rígido a 0.60 permissivo). Menor = mais difícil confundir duas pessoas parecidas.
    tempo_celular_seg: "4",  // cooldown (segundos) entre alertas de celular na sala — a detecção em si continua rápida
    tempo_distracao_seg: "5",// segundos de cabeça baixa até marcar "distraído"
    confirmacoes_celular: "2", // nº de detecções numa janela de ~1,2s para confirmar um celular antes do alerta (evita falso positivo isolado, ex: capa de celular)
    max_distancia_px: "800",  // distância máxima (pixels) entre celular e rosto mais próximo, só para EXIBIR de quem é o celular na tela (não bloqueia o alerta)
    acao_sensibilidade: "10"  // escala de 1 (mais seletiva) a 10 (mais sensível)
};
app.get("/config-ia", authMiddleware, adminMiddleware, async (req, res) => {
    const rows = await allQuery("SELECT chave, valor FROM configuracoes_ia");
    const config = { ...CONFIG_IA_DEFAULTS };
    rows.forEach(r => config[r.chave] = r.valor);
    res.json(config);
});
app.post("/config-ia", authMiddleware, adminMiddleware, async (req, res) => {
    if (req.body.acao_sensibilidade !== undefined) {
        const sensibilidade = Number(req.body.acao_sensibilidade);
        if (!Number.isInteger(sensibilidade) || sensibilidade < 1 || sensibilidade > 10) {
            return res.status(400).json({ error: "A sensibilidade da ação suspeita deve ser um número inteiro entre 1 e 10." });
        }
    }
    for (const chave of Object.keys(CONFIG_IA_DEFAULTS)) {
        if (req.body[chave] !== undefined) {
            await runQuery("INSERT INTO configuracoes_ia (chave, valor) VALUES (?, ?) ON CONFLICT(chave) DO UPDATE SET valor=excluded.valor", [chave, String(req.body[chave])]);
        }
    }
    res.json({ status: "ok" });
});


app.get("/alunos", authMiddleware, async (req, res) => res.json(await allQuery("SELECT a.matricula, a.nome, t.nome as turma_nome FROM alunos a JOIN turmas t ON a.turma_id = t.id WHERE a.usuario_id = ?", [req.user.id])));

app.post("/alunos", authMiddleware, async (req, res) => {
    try {
        const matricula = matriculaSegura(req.body.matricula);
        if (!matricula) return res.status(400).json({ error: "Matrícula inválida (use apenas letras, números, - e _)." });
        await runQuery("INSERT OR REPLACE INTO alunos (matricula, nome, turma_id, usuario_id) VALUES (?, ?, ?, ?)", [matricula, req.body.nome.trim(), req.body.turma_id, req.user.id]);
        if (req.body.foto_base64 && req.body.foto_base64.trim() !== "") {
            const base64Data = req.body.foto_base64.replace(/^data:image\/\w+;base64,/, "");
            fs.writeFileSync(path.resolve(__dirname, `../data/faces/${matricula}.jpg`), base64Data, 'base64');
            try { await axios.post(`${PYTHON_IA_URL}/api/reload_faces`); } catch(e) {} 
        }
        res.json({ status: "ok" });
    } catch (e) { res.status(500).json({ error: e.message }); }
});

app.post("/alunos/:id/foto", authMiddleware, async (req, res) => {
    try {
        const matricula = matriculaSegura(req.params.id);
        if (!matricula) return res.status(400).json({ error: "Matrícula inválida." });
        // Corrige IDOR: sem isso, qualquer professor logado podia sobrescrever a foto
        // de reconhecimento facial de um aluno de OUTRO professor, só sabendo a matrícula.
        const aluno = await getQuery("SELECT matricula FROM alunos WHERE matricula = ? AND usuario_id = ?", [matricula, req.user.id]);
        if (!aluno) return res.status(404).json({ error: "Aluno não encontrado ou não pertence a você." });
        if (req.body.foto_base64 && req.body.foto_base64.trim() !== "") {
            const base64Data = req.body.foto_base64.replace(/^data:image\/\w+;base64,/, "");
            fs.writeFileSync(path.resolve(__dirname, `../data/faces/${matricula}.jpg`), base64Data, 'base64');
            try { await axios.post(`${PYTHON_IA_URL}/api/reload_faces`); } catch(e) {} 
            res.json({ status: "ok" });
        } else {
            res.status(400).json({ error: "Foto não fornecida" });
        }
    } catch (e) { res.status(500).json({ error: e.message }); }
});

app.delete("/alunos/:id", authMiddleware, async (req, res) => {
    const matricula = matriculaSegura(req.params.id);
    if (!matricula) return res.status(400).json({ error: "Matrícula inválida." });
    await runQuery("DELETE FROM alunos WHERE matricula=? AND usuario_id=?", [matricula, req.user.id]);
    const filePath = path.resolve(__dirname, `../data/faces/${matricula}.jpg`);
    if(fs.existsSync(filePath)) fs.unlinkSync(filePath);
    try { await axios.post(`${PYTHON_IA_URL}/api/reload_faces`); } catch(e) {}
    res.json({ status: "ok" });
});

app.get("/materias/:id/alunos", authMiddleware, async (req, res) => res.json(await allQuery("SELECT a.matricula, a.nome FROM alunos a JOIN turmas t ON a.turma_id = t.id JOIN materias m ON m.turma_id = t.id WHERE m.id = ? AND a.usuario_id = ?", [req.params.id, req.user.id])));

app.get("/assentos/:materia_id/:sala_id", authMiddleware, async (req, res) => res.json(await allQuery("SELECT aluno_matricula, bancada_nome FROM assentos WHERE materia_id = ? AND sala_id = ?", [req.params.materia_id, req.params.sala_id])));
app.post("/assentos", authMiddleware, async (req, res) => {
    await runQuery("DELETE FROM assentos WHERE materia_id = ? AND sala_id = ?", [req.body.materia_id, req.body.sala_id]);
    for(let a of req.body.assentos) await runQuery("INSERT INTO assentos (aluno_matricula, materia_id, sala_id, bancada_nome) VALUES (?, ?, ?, ?)", [a.aluno_matricula, req.body.materia_id, req.body.sala_id, a.bancada_nome]);
    res.json({ status: "ok" });
});

app.get("/aula/status", authMiddleware, async (req, res) => res.json(await getQuery(`SELECT a.id, a.materia_id, a.hora_inicio, COALESCE(m.nome, 'Matéria removida') as materia_nome, COALESCE(t.nome, '') as turma_nome, COALESCE(s.nome, 'Sala removida') as sala_nome, s.camera_url, s.id as sala_id FROM aulas a LEFT JOIN materias m ON a.materia_id = m.id LEFT JOIN turmas t ON m.turma_id = t.id LEFT JOIN salas s ON a.sala_id = s.id WHERE a.status = "EM_ANDAMENTO" AND a.usuario_id = ? ORDER BY a.id DESC LIMIT 1`, [req.user.id]) || null));
app.post("/aula/iniciar", authMiddleware, async (req, res) => {
    const ativo = await getQuery(`SELECT id FROM aulas WHERE status = "EM_ANDAMENTO" AND usuario_id = ?`, [req.user.id]);
    if(ativo) return res.status(400).json({ error: "Você já tem uma aula em andamento. Encerre-a antes de iniciar outra." });
    // Trava multiusuário: verifica se ALGUÉM (qualquer professor) já está usando essa sala/câmera.
    const ocupada = await getQuery(`SELECT a.id, u.nome as professor_nome, m.nome as materia_nome, a.hora_inicio FROM aulas a JOIN usuarios u ON a.usuario_id = u.id JOIN materias m ON a.materia_id = m.id WHERE a.status = "EM_ANDAMENTO" AND a.sala_id = ?`, [req.body.sala_id]);
    if(ocupada) return res.status(409).json({ error: `Sala ocupada por ${ocupada.professor_nome} (${ocupada.materia_nome}) desde ${ocupada.hora_inicio}. Aguarde ela encerrar a aula.` });
    const dia = new Date().toISOString().split('T')[0]; const hora = new Date().toTimeString().split(' ')[0].substring(0,8);
    const result = await runQuery(`INSERT INTO aulas (materia_id, sala_id, data_dia, hora_inicio, status, usuario_id) VALUES (?, ?, ?, ?, "EM_ANDAMENTO", ?)`, [req.body.materia_id, req.body.sala_id, dia, hora, req.user.id]);
    // live_status agora é isolado por aula_id — nenhuma limpeza global é necessária,
    // uma aula nova nunca terá linhas de status residuais de outra aula.
    res.json({ status: "ok" });
});
app.post("/aula/encerrar", authMiddleware, async (req, res) => {
    const hora = new Date().toTimeString().split(' ')[0].substring(0,8);
    const aulaAtiva = await getQuery(`SELECT id FROM aulas WHERE status="EM_ANDAMENTO" AND usuario_id=?`, [req.user.id]);
    await runQuery(`UPDATE aulas SET status="FINALIZADA", hora_fim=? WHERE status="EM_ANDAMENTO" AND usuario_id=?`, [hora, req.user.id]);
    if (aulaAtiva) await runQuery(`DELETE FROM live_status WHERE aula_id = ?`, [aulaAtiva.id]); // limpa só a live_status DESTA aula, nunca a de outro professor
    res.json({ status: "ok" });
});

// Status de todas as salas: livre, ou ocupada por quem/qual matéria/desde quando.
// Usado para bloquear a UI antes mesmo do professor tentar iniciar a aula.
app.get("/salas/status", authMiddleware, async (req, res) => {
    const salas = await allQuery(`SELECT id, nome FROM salas`);
    const ocupadas = await allQuery(`SELECT a.sala_id, u.nome as professor_nome, m.nome as materia_nome, a.hora_inicio FROM aulas a JOIN usuarios u ON a.usuario_id = u.id JOIN materias m ON a.materia_id = m.id WHERE a.status = "EM_ANDAMENTO"`);
    const mapaOcupadas = {}; ocupadas.forEach(o => mapaOcupadas[o.sala_id] = o);
    res.json(salas.map(s => ({ ...s, ocupada: !!mapaOcupadas[s.id], detalhes: mapaOcupadas[s.id] || null })));
});


app.get("/aula/status_alunos", authMiddleware, async (req, res) => {
    // As duas consultas são independentes: dispara a de live_status junto, em vez de esperar a outra terminar.
    const livePromise = allQuery(`SELECT * FROM live_status WHERE aula_id = (SELECT id FROM aulas WHERE status = "EM_ANDAMENTO" AND usuario_id = ? ORDER BY id DESC LIMIT 1)`, [req.user.id]);
    const rows = await allQuery(`SELECT p1.aluno_id, al.nome as aluno_nome, p1.bancada_atual, p1.data_hora, p1.tipo, ast.bancada_nome as bancada_fixa FROM presenca p1 JOIN alunos al ON p1.aluno_id = al.matricula INNER JOIN (SELECT aluno_id, MAX(id) as max_id FROM presenca WHERE aula_id = (SELECT id FROM aulas WHERE status = "EM_ANDAMENTO" AND usuario_id = ? ORDER BY id DESC LIMIT 1) GROUP BY aluno_id) p2 ON p1.id = p2.max_id LEFT JOIN aulas a ON p1.aula_id = a.id LEFT JOIN assentos ast ON p1.aluno_id = ast.aluno_matricula AND a.materia_id = ast.materia_id AND a.sala_id = ast.sala_id`, [req.user.id]);
    const live = await livePromise;
    let result = [];
    for(let r of rows) {
        const stat_row = live.find(l => String(l.aluno_matricula) === String(r.aluno_id));
        // A bancada/status "ao vivo" vêm do live_status (atualizado a cada ciclo da IA);
        // a tabela presenca só é gravada a cada ~10s e servia de fallback.
        result.push({ ...r, status_atencao: stat_row ? stat_row.status_atencao : "FOCADO", bancada_atual: (stat_row && stat_row.bancada_atual) ? stat_row.bancada_atual : r.bancada_atual });
    }
    res.json(result);
});

app.get("/aulas/historico", authMiddleware, async (req, res) => res.json(await allQuery(`SELECT a.id, a.data_dia, a.hora_inicio, a.hora_fim, m.nome as materia_nome, t.nome as turma_nome, s.nome as sala_nome, a.status FROM aulas a JOIN materias m ON a.materia_id = m.id JOIN turmas t ON m.turma_id = t.id JOIN salas s ON a.sala_id = s.id WHERE a.usuario_id = ? ORDER BY a.id DESC LIMIT 50`, [req.user.id])));

app.get("/alertas", authMiddleware, async (req, res) => res.json(await allQuery(`SELECT a.id, a.mensagem, a.data_hora, a.video_file, a.image_file, m.nome as materia_nome, s.nome as sala_nome FROM alertas a JOIN aulas au ON a.aula_id = au.id JOIN materias m ON au.materia_id = m.id JOIN salas s ON au.sala_id = s.id WHERE au.usuario_id = ? ORDER BY a.id DESC LIMIT 100`, [req.user.id])));

// Rota /justificar removida: abonar falta desativado (não é permitido).

app.get("/analytics", authMiddleware, async (req, res) => {
    const total_alunos = (await getQuery("SELECT COUNT(*) as c FROM alunos WHERE usuario_id = ?", [req.user.id])).c;
    const total_aulas = (await getQuery("SELECT COUNT(*) as c FROM aulas WHERE usuario_id = ?", [req.user.id])).c;
    const total_alertas = (await getQuery("SELECT COUNT(*) as c FROM alertas a JOIN aulas au ON a.aula_id = au.id WHERE au.usuario_id = ?", [req.user.id])).c;
    const freq_turmas = await allQuery(`
        SELECT t.nome, 
               (SELECT COUNT(*) FROM alunos WHERE turma_id = t.id) as matriculados,
               (SELECT COUNT(DISTINCT p.aluno_id) FROM presenca p JOIN aulas a ON p.aula_id = a.id JOIN materias m ON a.materia_id = m.id WHERE m.turma_id = t.id) as presencas
        FROM turmas t WHERE t.usuario_id = ?
    `, [req.user.id]);
    res.json({ total_alunos, total_aulas, total_alertas, frequencia_turmas: freq_turmas });
});

app.get("/analytics/retencao", authMiddleware, async (req, res) => {
    let foco_pct = 100; let cel_pct = 0; let disp_pct = 0;
    const aula = await getQuery('SELECT id FROM aulas WHERE usuario_id = ? ORDER BY id DESC LIMIT 1', [req.user.id]);
    if(aula) {
        const alrts = await allQuery("SELECT mensagem FROM alertas WHERE aula_id = ?", [aula.id]);
        let cel_count = alrts.filter(a => a.mensagem.includes("CELULAR")).length;
        let disp_count = alrts.filter(a => a.mensagem.includes("DISPERSÃO") || a.mensagem.includes("AÇÃO SUSPEITA")).length;
        if (cel_count > 0 || disp_count > 0) {
            cel_pct = Math.min(cel_count * 5, 100); 
            disp_pct = Math.min(disp_count * 3, 100 - cel_pct); 
            foco_pct = Math.max(100 - cel_pct - disp_pct, 0);
        }
    }
    res.json({ labels: ["Focado (%)", "Uso de Celular (%)", "Dispersão/Distração (%)"], data: [foco_pct, cel_pct, disp_pct] });
});

app.get("/aula/:id/relatorio", authMiddleware, async (req, res) => {
    try {
        let aulaId = req.params.id;
        if (aulaId === 'last') {
            const lastAula = await getQuery('SELECT id FROM aulas WHERE usuario_id = ? ORDER BY id DESC LIMIT 1', [req.user.id]);
            if(!lastAula) return res.status(404).send("Nenhuma aula encontrada para gerar relatório.");
            aulaId = lastAula.id;
        }

        // Corrige IDOR: sem "AND a.usuario_id = ?" abaixo, qualquer professor logado
        // conseguia baixar o relatório de presença de uma aula de OUTRO professor
        // apenas trocando o número na URL.
        const aula = await getQuery("SELECT a.id, a.data_dia, a.hora_inicio, a.hora_fim, m.nome as materia_nome FROM aulas a JOIN materias m ON a.materia_id = m.id WHERE a.id = ? AND a.usuario_id = ?", [aulaId, req.user.id]);
        if(!aula) return res.status(404).send("Aula não encontrada ou não pertence a você.");

        const alertas = await allQuery("SELECT mensagem, data_hora FROM alertas WHERE aula_id=?", [aulaId]);
        const presencas = await allQuery("SELECT al.matricula, al.nome, MIN(p.data_hora) as primeiro_registro, MAX(p.data_hora) as ultimo_registro, COUNT(p.id) as eventos FROM presenca p JOIN alunos al ON p.aluno_id = al.matricula WHERE p.aula_id = ? GROUP BY p.aluno_id", [aulaId]);

        const workbook = new ExcelJS.Workbook(); const sheet = workbook.addWorksheet('Diário de Classe');
        sheet.mergeCells('A1:G1'); const titulo = sheet.getCell('A1'); 
        titulo.value = `DIÁRIO DE CLASSE - ${aula.materia_nome ? aula.materia_nome.toUpperCase() : "MATÉRIA"} - DATA: ${aula.data_dia}`; 
        titulo.font = { size: 14, bold: true }; titulo.alignment = { horizontal: 'center' };
        
        sheet.addRow(['MATRÍCULA', 'NOME DO ALUNO', 'ENTRADA', 'SAÍDA', 'TEMPO (min)', 'STATUS FINAL', 'OBSERVAÇÕES DA IA']);
        sheet.getRow(2).eachCell(c => { c.fill = {type: 'pattern', pattern:'solid', fgColor:{argb:'1E293B'}}; c.font = {color:{argb:'FFFFFF'}, bold:true}; c.alignment = {horizontal:'center'}; });

        presencas.forEach(p => {
            let status = "PRESENTE"; let tempo = 0;
            if(p.primeiro_registro && p.ultimo_registro) tempo = Math.round((new Date(p.ultimo_registro) - new Date(p.primeiro_registro)) / 60000);
            
            let obs = []; 
            alertas.forEach(a => { 
                if(a.mensagem.includes(p.matricula) || a.mensagem.includes(p.nome)) { 
                    let h = a.data_hora.split(" ")[1]; 
                    if(a.mensagem.includes("CELULAR")) obs.push(`Celular (${h})`); 
                    if(a.mensagem.includes("DISPERSÃO")) obs.push(`Distraído (${h})`); 
                    if(a.mensagem.includes("AÇÃO SUSPEITA")) obs.push(`Ação suspeita (${h})`); 
                } 
            });
            
            const row = sheet.addRow([p.matricula, p.nome, p.primeiro_registro, p.ultimo_registro, tempo, status, obs.join(' | ') || "Sem ocorrências."]);
            row.eachCell((c, i) => { c.alignment = (i===2||i===7) ? {horizontal:'left'} : {horizontal:'center'}; });
            const statCell = row.getCell(6); statCell.font = {color:{argb:'FFFFFF'}, bold:true};
            if(status.includes("JUSTIFICADO")) statCell.fill = {type:'pattern', pattern:'solid', fgColor:{argb:'3B82F6'}}; else if(status === "PRESENTE") statCell.fill = {type:'pattern', pattern:'solid', fgColor:{argb:'7AC142'}}; else statCell.fill = {type:'pattern', pattern:'solid', fgColor:{argb:'E30613'}};
        });

        sheet.columns.forEach(col => col.width = 25); sheet.getColumn(7).width = 45;
        
        sheet.addRow([]);
        const sumRow1 = sheet.addRow(['📊 RELATÓRIO COMPORTAMENTAL IA']);
        sumRow1.font = { bold: true, color: { argb: 'FFFFFF' } };
        sumRow1.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: '1E293B' } };
        sheet.mergeCells(`A${sumRow1.number}:G${sumRow1.number}`);
        
        let cel_count = alertas.filter(a => a.mensagem.includes("CELULAR")).length;
        let disp_count = alertas.filter(a => a.mensagem.includes("DISPERSÃO") || a.mensagem.includes("AÇÃO SUSPEITA")).length;
        let cel_pct = Math.min(cel_count * 5, 100);
        let disp_pct = Math.min(disp_count * 3, 100 - cel_pct);
        let foco_pct = Math.max(100 - cel_pct - disp_pct, 0);

        sheet.addRow(['Foco Geral da Turma:', `${foco_pct}%`]);
        sheet.addRow(['Índice de Uso de Celular:', `${cel_pct}%`]);
        sheet.addRow(['Índice de Dispersão:', `${disp_pct}%`]);
        
        const buffer = await workbook.xlsx.writeBuffer();
        res.setHeader('Content-Type', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'); 
        res.setHeader('Content-Disposition', `attachment; filename="DiarioClasse.xlsx"`);
        res.send(buffer);
    } catch(e) { console.error("Erro Excel:", e); res.status(500).send("Erro interno ao gerar planilha."); }
});

app.listen(3000, () => console.log("🟢 API Administrativa Node.js rodando na Porta 3000"));