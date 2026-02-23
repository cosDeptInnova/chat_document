/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class', // Habilitar dark mode con clase
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#f0f7ff',
          100: '#e0efff',
          200: '#b9dfff',
          300: '#7cc5ff',
          400: '#36a8ff',
          500: '#0d8fff',
          600: '#006ee6',
          700: '#0057b3',
          800: '#054a93',
          900: '#0a3f78',
        },
        cosmos: {
          blue: '#0066FF', // Azul vibrante de Cosmos
        },
      },
    },
  },
  plugins: [],
}

