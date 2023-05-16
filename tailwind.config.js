/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{vue,js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        primary: '#9B7AA0',
        dark: '#210B2C',
        'dark-50': '#877f90',
        base: '#EDF2F4',
        error: '#D90429',
      },
    },
  },
  plugins: [],
};