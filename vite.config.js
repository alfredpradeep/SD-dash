import { defineConfig } from 'vite';
import { spawn } from 'child_process';

function pythonBackend() {
  let proc;
  return {
    name: 'python-backend',
    configureServer() {
      proc = spawn('python3', ['-m', 'uvicorn', 'compress.main:app', '--host', '0.0.0.0', '--port', '8001', '--log-level', 'info'], {
        env: { ...process.env, COMPRESS_DEV_MODE: '1', PATH: `${process.env.HOME}/.local/bin:${process.env.PATH}` },
        stdio: 'inherit',
      });
      proc.on('error', (err) => console.error('Backend failed:', err.message));
    },
    closeBundle() {
      if (proc) proc.kill();
    },
  };
}

export default defineConfig({
  plugins: [pythonBackend()],
  server: {
    proxy: {
      '/compress': 'http://localhost:8001',
      '/v4': 'http://localhost:8001',
      '/static': 'http://localhost:8001',
      '/docs': 'http://localhost:8001',
    },
  },
});
