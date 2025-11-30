/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx}",
    "./pages/**/*.{js,ts,jsx,tsx}",
    "./components/**/*.{js,ts,jsx,tsx}",
  ],

  darkMode: "class",

  theme: {
    extend: {
      fontFamily: {
        inter: ["Inter", "sans-serif"],
      },

      /* -----------------------------------
         NEON COLOR PALETTE
      ----------------------------------- */
      colors: {
        neon: {
          purple: "#b54fff",
          pink: "#ff44cc",
          blue: "#4fd6ff",
        },
        background: "#050007",
        purple900: "#0b0220",
      },

      /* -----------------------------------
         SHADOWS — Glass + Glow
      ----------------------------------- */
      boxShadow: {
        glow: "0 0 25px rgba(180, 0, 255, 0.35)",
        neon: "0 0 50px rgba(255, 20, 180, 0.45)",
        card: "0 8px 32px rgba(0,0,0,0.45)",
        insetGlow:
          "inset 0 0 20px rgba(180,0,255,0.25), 0 0 20px rgba(180,0,255,0.2)",
      },

      /* -----------------------------------
         BACKDROP FILTERS
      ----------------------------------- */
      backdropBlur: {
        xs: "2px",
        sm: "4px",
        md: "10px",
        lg: "14px",
        xl: "18px",
      },

      /* -----------------------------------
         ANIMATIONS & KEYFRAMES
      ----------------------------------- */
      keyframes: {
        float: {
          "0%, 100%": { transform: "translateY(0px)" },
          "50%": { transform: "translateY(-6px)" },
        },
        gradientMove: {
          "0%": { backgroundPosition: "0% 50%" },
          "50%": { backgroundPosition: "100% 50%" },
          "100%": { backgroundPosition: "0% 50%" },
        },
        glowPulse: {
          "0%": { boxShadow: "0 0 10px rgba(168, 85, 247, 0.3)" },
          "50%": { boxShadow: "0 0 20px rgba(168, 85, 247, 0.6)" },
          "100%": { boxShadow: "0 0 10px rgba(168, 85, 247, 0.3)" },
        },
      },

      animation: {
        float: "float 4s ease-in-out infinite",
        gradientMove: "gradientMove 5s ease infinite",
        glowPulse: "glowPulse 2s ease-in-out infinite",
      },

      /* -----------------------------------
         RADIUS
      ----------------------------------- */
      borderRadius: {
        glass: "20px",
        big: "28px",
      },

      /* -----------------------------------
         TRANSITIONS
      ----------------------------------- */
      transitionTimingFunction: {
        smooth: "cubic-bezier(0.4, 0.0, 0.2, 1)",
      },
    },
  },

  plugins: [],
};
