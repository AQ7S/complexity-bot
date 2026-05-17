export default [
  {
    ignores: [
      'dist/**',
      'dist-electron/**',
      'release/**',
      'node_modules/**',
      '**/*.d.ts',
      '**/*.ts',
      '**/*.tsx',
    ],
  },
  {
    files: ['**/*.{js,jsx,mjs,cjs}'],
    languageOptions: {
      ecmaVersion: 'latest',
      sourceType: 'module',
      globals: {
        window: 'readonly', document: 'readonly', navigator: 'readonly',
        console: 'readonly', setTimeout: 'readonly', clearTimeout: 'readonly',
        setInterval: 'readonly', clearInterval: 'readonly', requestAnimationFrame: 'readonly',
        cancelAnimationFrame: 'readonly', localStorage: 'readonly', sessionStorage: 'readonly',
        fetch: 'readonly', WebSocket: 'readonly', URL: 'readonly', URLSearchParams: 'readonly',
        Blob: 'readonly', File: 'readonly', FileReader: 'readonly',
        process: 'readonly', global: 'readonly', Buffer: 'readonly', __dirname: 'readonly',
        __filename: 'readonly', module: 'readonly', require: 'readonly', exports: 'readonly',
      },
      parserOptions: {
        ecmaFeatures: { jsx: true },
      },
    },
    rules: {
      'no-undef': 'off',
      'no-unused-vars': 'off',
      'no-empty': ['warn', { allowEmptyCatch: true }],
      'no-constant-condition': ['warn', { checkLoops: false }],
      'no-useless-escape': 'warn',
    },
  },
];
