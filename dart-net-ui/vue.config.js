// vue.config.js
module.exports = {
    devServer: {
        proxy: {
          '^/nets': {
            target: 'http://localhost:8080',
            changeOrigin: true
          },
          '^/foo': {
            target: '<other_url>'
          }
        }
      }
}