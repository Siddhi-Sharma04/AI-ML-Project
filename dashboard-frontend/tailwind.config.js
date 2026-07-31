/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
    "./components/**/*.{js,ts,jsx,tsx}",
    "./pages/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        darkBg: "#0B0F19",
        cardBg: "rgba(17, 24, 39, 0.7)",
        primaryBlue: "#1E3A8A",
        accentNeon: "#3B82F6",
        violationRed: "#EF4444",
        paidGreen: "#10B981",
        warningAmber: "#F59E0B"
      },
      fontFamily: {
        sans: ["Inter", "sans-serif"]
      },
      backdropBlur: {
        xs: "2px"
      }
    },
  },
  plugins: [],
}
