require('dotenv').config();
const express = require('express');
const path = require('path');
const multer = require('multer');
const axios = require('axios');
const FormData = require('form-data');
const { spawn } = require('child_process');

const app = express();
const upload = multer({ storage: multer.memoryStorage() });

const PORT = process.env.PORT || 3000;
const PYTHON_URL = process.env.PYTHON_API_URL;

const pythonProcess = spawn('python', ['main.py']);
pythonProcess.stdout.on('data', (data) => console.log(`Python: ${data}`));
pythonProcess.stderr.on('data', (data) => console.error(`Python: ${data}`));
process.on('exit', () => pythonProcess.kill());

app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(express.static(path.join(__dirname, 'public')));

// ── Pages ──────────────────────────────────────────────────────────────────

app.get('/', (req, res) => res.render('index'));

// ── History & Stats (proxied from Python/DB) ───────────────────────────────

app.get('/history', async (req, res) => {
    try {
        const limit = req.query.limit || 20;
        const response = await axios.get(`${PYTHON_URL}/history?limit=${limit}`);
        res.json(response.data);
    } catch (error) {
        res.status(500).json({ error: error.response?.data?.detail || error.message });
    }
});

app.delete('/history', async (req, res) => {
    try {
        const response = await axios.delete(`${PYTHON_URL}/history`);
        res.json(response.data);
    } catch (error) {
        res.status(500).json({ error: error.response?.data?.detail || error.message });
    }
});

app.get('/stats', async (req, res) => {
    try {
        const response = await axios.get(`${PYTHON_URL}/stats`);
        res.json(response.data);
    } catch (error) {
        res.status(500).json({ error: error.response?.data?.detail || error.message });
    }
});

// ── CSV Export ─────────────────────────────────────────────────────────────

app.get('/export/csv', async (req, res) => {
    try {
        const limit = req.query.limit || 100;
        const response = await axios.get(`${PYTHON_URL}/export/csv?limit=${limit}`, {
            responseType: 'stream'
        });
        res.setHeader('Content-Type', 'text/csv');
        res.setHeader('Content-Disposition', 'attachment; filename=crowd_analysis.csv');
        response.data.pipe(res);
    } catch (error) {
        res.status(500).json({ error: error.response?.data?.detail || error.message });
    }
});

// ── Upload Image ───────────────────────────────────────────────────────────

app.post('/upload', upload.single('file'), async (req, res) => {
    try {
        if (!req.file) return res.status(400).json({ error: 'No file uploaded' });

        const formData = new FormData();
        formData.append('file', req.file.buffer, req.file.originalname);
        const threshold = req.query.threshold || 10;

        const response = await axios.post(`${PYTHON_URL}/upload?threshold=${threshold}`, formData, {
            headers: formData.getHeaders(),
            maxContentLength: Infinity,
            maxBodyLength: Infinity
        });

        res.json(response.data);
    } catch (error) {
        res.status(500).json({ error: error.response?.data?.detail || error.message });
    }
});

// ── Upload Video ───────────────────────────────────────────────────────────

app.post('/upload-video', upload.single('file'), async (req, res) => {
    try {
        if (!req.file) return res.status(400).json({ error: 'No file uploaded' });

        const formData = new FormData();
        formData.append('file', req.file.buffer, {
            filename: req.file.originalname,
            contentType: req.file.mimetype
        });

        const response = await axios.post(`${PYTHON_URL}/upload-video`, formData, {
            headers: formData.getHeaders(),
            maxContentLength: Infinity,
            maxBodyLength: Infinity,
            timeout: 300000
        });

        res.json(response.data);
    } catch (error) {
        res.status(500).json({ error: error.response?.data?.detail || error.message });
    }
});

// ── Webcam Frame ───────────────────────────────────────────────────────────

app.post('/webcam-frame', upload.single('file'), async (req, res) => {
    try {
        if (!req.file) return res.status(400).json({ error: 'No frame received' });

        const formData = new FormData();
        formData.append('file', req.file.buffer, {
            filename: 'frame.jpg',
            contentType: 'image/jpeg'
        });
        const threshold = req.query.threshold || 10;

        const response = await axios.post(`${PYTHON_URL}/webcam-frame?threshold=${threshold}`, formData, {
            headers: formData.getHeaders(),
            maxContentLength: Infinity,
            maxBodyLength: Infinity,
            timeout: 15000
        });

        res.json(response.data);
    } catch (error) {
        res.status(500).json({ error: error.response?.data?.detail || error.message });
    }
});

// ── Start ──────────────────────────────────────────────────────────────────

function waitForPython(retries = 15) {
    axios.get(`${PYTHON_URL}/health`)
        .then(() => {
            app.listen(PORT, () => {
                console.log(`\n✅ Express Server: http://localhost:${PORT}`);
                console.log(`✅ Python API:     ${PYTHON_URL}`);
                console.log(`\n👉 Open http://localhost:${PORT} in your browser\n`);
            });
        })
        .catch(() => {
            if (retries > 0) {
                console.log('Waiting for Python server...');
                setTimeout(() => waitForPython(retries - 1), 2000);
            } else {
                console.error('Python server failed to start.');
                process.exit(1);
            }
        });
}

waitForPython();
