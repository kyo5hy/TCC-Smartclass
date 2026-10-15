// ecosystem.config.js
module.exports = {
  apps: [
    {
      name: "smartclass-node",
      script: "./node_api/server.js",
      watch: false,
      env: {
        NODE_ENV: "production",
      },
      error_file: "./logs/node-error.log",
      out_file: "./logs/node-out.log",
      time: true
    },
    {
      name: "smartclass-ia",
      script: "./python_ia/main.py",
      interpreter: "./python_ia/venv/bin/python",
      watch: false,
      env: {
        PYTHONUNBUFFERED: "1",
      },
      error_file: "./logs/python-error.log",
      out_file: "./logs/python-out.log",
      time: true
    }
  ]
};