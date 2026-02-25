import React from "react";

export default function PasswordField({
  label,
  value,
  onChange,
  showPassword,
  toggleShowPassword,
  isDarkMode = false,
  autoComplete = "new-password",
}) {
  return (
    <div className="input-group w-full px-4 md:px-0">
      {/* Label con candado al lado */}
      <div className="flex items-center mb-1.5">
        <label className="block text-sm font-medium">{label}</label>
      </div>

      {/* Input con botón ojo */}
      <div className="relative">
        <input
          type={showPassword ? "text" : "password"}
          placeholder={label}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          autoComplete={autoComplete}
          spellCheck="false"
          className={`w-full pr-12 p-3 border rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors duration-200
            ${
              isDarkMode
                ? "bg-gray-800 text-white border-gray-700"
                : "bg-white text-black border-gray-300"
            }
          `}
          required
        />

        {/* Botón ojo dentro del input */}
        <button
          type="button"
          onClick={toggleShowPassword}
          className={`
            absolute right-3 top-1/2 -translate-y-1/2
            flex items-center justify-center
            transition-transform duration-300
            ${
              showPassword
                ? "rotate-180 scale-110 text-blue-500"
                : "rotate-0 scale-100 text-gray-500 hover:text-blue-400"
            }
          `}
          style={{ background: "none", border: "none", padding: 0, cursor: "pointer" }}
        >
          {showPassword ? (
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-5 w-5"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={2}
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              <path strokeLinecap="round" strokeLinejoin="round" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.477 0 8.268 2.943 9.542 7-1.274 4.057-5.065 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
            </svg>
          ) : (
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-5 w-5"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={2}
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="M13.875 18.825A10.05 10.05 0 0112 19c-4.477 0-8.268-2.943-9.542-7a10.05 10.05 0 012.354-3.431m3.732-2.472A9.967 9.967 0 0112 5c4.477 0 8.268 2.943 9.542 7a10.05 10.05 0 01-1.372 2.424M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              <path strokeLinecap="round" strokeLinejoin="round" d="M3 3l18 18" />
            </svg>
          )}
        </button>
      </div>
    </div>
  );
}