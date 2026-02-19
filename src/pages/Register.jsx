import React, { useState, useMemo } from "react";
import { useNavigate, Link } from "react-router-dom";
import AuthLayout from "../layouts/AuthLayout";
import PasswordField from "../components/utils/PasswordField";
import RegisterErrorModal from "../components/utils/RegisterErrorModal";
import "../styles/register.css";
import cosmosLogo from "../images/cosmosLogo.png";

export default function Register() {
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [showErrorModal, setShowErrorModal] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [emailError, setEmailError] = useState(""); // Estado para gestionar que el correo solo pueda llevar @cosgs.com como extensión
  const navigate = useNavigate();

  const [department, setDepartment] = useState("");
  const [role, setRole] = useState("");

  const handleEmailChange = (e) => {
    const value = e.target.value;
    setEmail(value);

    // Validación en vivo
    if (value && !value.endsWith("@cosgs.com")) {
      setEmailError("El correo debe terminar en @cosgs.com");
    } else {
      setEmailError("");
    }
  };

  // Validaciones individuales
  const passwordChecks = useMemo(() => ({
    length: password.length >= 8 && password.length <= 12,
    lowercase: /[a-z]/.test(password),
    uppercase: /[A-Z]/.test(password),
    number: /\d/.test(password),
    symbol: /[@$!%*?&.,;:_\-]/.test(password),
  }), [password]);

  const allValid = Object.values(passwordChecks).every(Boolean);
  const passwordsMatch = password === confirmPassword;

  const handleSubmit = (e) => {
    e.preventDefault();

    // Validación para la extensión del correo antes de enviar para que sea @cosgs.com
    if (!email.endsWith("@cosgs.com")) {
      setErrorMessage("El correo debe terminar en @cosgs.com");
      setShowErrorModal(true);
      return;
    }

    // Validación para el departamento, que no esté vacío
    if (!department) {
      setErrorMessage("Debes seleccionar un departamento para el usuario.");
      setShowErrorModal(true);
      return;
    }

    // Validación para el rol, que no esté vacío
    if (!role) {
      setErrorMessage("Debes seleccionar un rol.");
      setShowErrorModal(true);
      return;
    }
    
    // Esto que en principio nunca se usará porque mientras la contraseña no cumpla los requisitos de seguridad, el boton de registrar estará desactivado
    if (!allValid) {
      setErrorMessage("La contraseña no cumple con todos los requisitos de seguridad.");
      setShowErrorModal(true);
      return;
    }
    
    if (!passwordsMatch) {
      setErrorMessage("Las contraseñas no coinciden. Por favor, verifica tu contraseña.");
      setShowErrorModal(true);
      return;
    }

    const createdAt = new Date().toISOString();
    const newUser = {
      name: fullName,
      email,
      password,
      department,
      role,
      createdAt,
    };

    localStorage.setItem("registeredUser", JSON.stringify(newUser));
    console.log("Usuario registrado:", newUser);

    // Mostrar toast de éxito
    const toast = document.createElement("div");
    toast.className = "fixed bottom-6 right-6 z-50 animate-fade-in";
    toast.innerHTML = `
      <div class="bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded-lg shadow-lg relative transition-all duration-500">
        <div class="flex items-center">
          <div class="py-1"><svg class="fill-current h-6 w-6 text-green-500 mr-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20"><path d="M2.93 17.07A10 10 0 1 1 17.07 2.93 10 10 0 0 1 2.93 17.07zm12.73-1.41A8 8 0 1 0 4.34 4.34a8 8 0 0 0 11.32 11.32zM9 11V9h2v6H9v-4zm0-6h2v2H9V5z"/></svg></div>
          <div>
            <p class="font-bold">¡Registro exitoso!</p>
            <p class="text-sm">Tu cuenta ha sido creada correctamente.</p>
          </div>
        </div>
        <div class="absolute bottom-0 left-0 h-1 bg-green-500 animate-progress w-full"></div>
      </div>
    `;
    document.body.appendChild(toast);

    // Redirigir después de 3 segundos
    setTimeout(() => {
      toast.remove();
      navigate("/login", { replace: true });
    }, 3000);
  };

  return (
    <AuthLayout>
      <div className="register-wrapper">
        <div className="container relative z-10 flex flex-col gap-4">
          {/* Panel izquierdo */}
          <div className="flex">
            <div className="left-panel">
              <div className="logo">
                <img 
                  src={cosmosLogo} 
                  alt="Cosmos Logo" 
                  style={{ 
                    width: "200px", 
                    height: "70px", 
                    objectFit: "contain", 
                    marginRight: "10px" 
                  }} 
                />
                <h1>AI Assistant</h1>
              </div>
              <h3>Accede a la inteligencia artificial más avanzada</h3>
              <div className="features">
                <div className="feature">
                  <i className="fas fa-brain"></i>
                  <span>Procesamiento de lenguaje natural</span>
                </div>
                <div className="feature">
                  <i className="fas fa-bolt"></i>
                  <span>Respuestas en tiempo real</span>
                </div>
                <div className="feature">
                  <i className="fas fa-shield-alt"></i>
                  <span>Seguridad de nivel empresarial</span>
                </div>
                <div className="feature">
                  <i className="fas fa-sync-alt"></i>
                  <span>Aprendizaje continuo</span>
                </div>
              </div>
            </div>

            {/* Formulario de registro */}
            <div className="form-container">
              <h2>Crear Cuenta</h2>
              <p>Regístrate para comenzar a usar nuestra plataforma</p>
              
              <form onSubmit={handleSubmit}>
                {/* Nombre completo */}
                <div className="input-group">
                  <label>
                    <i className="fas fa-user"></i> Nombre completo
                  </label>
                  <input 
                    type="text" 
                    placeholder="Tu nombre completo" 
                    value={fullName} 
                    onChange={(e) => setFullName(e.target.value)} 
                    required 
                  />
                </div>

                {/* Email */}
                <div className="input-group">
                  <label>
                    <i className="fas fa-envelope"></i> Correo electrónico
                  </label>
                  <input 
                    type="email" 
                    placeholder="tucorreo@cosgs.com" 
                    value={email} 
                    onChange={handleEmailChange} 
                    required 
                    autoComplete="new-email"
                  />
                  {emailError && <p className="!text-xs !text-red-500 !mt-1">{emailError}</p>}
                </div>

                {/* Departamento */}
                <div className="input-group select-wrapper">
                  <label>Departamento al que perteneces</label>
                  <select
                    value={department}
                    onChange={(e) => setDepartment(e.target.value)}
                    required
                    className="input-like-select"
                  >
                    <option value="" disabled hidden>
                      Selecciona un departamento
                    </option>
                    <option value="Recursos Humanos">Recursos Humanos</option>
                    <option value="Administración">Administración</option>
                    <option value="Comunicación">Comunicación</option>
                    <option value="Comercial">Comercial</option>
                    <option value="Ventas">Ventas</option>
                    <option value="Almacén">Almacén</option>
                    <option value="Operaciones e Innovación">Operaciones e Innovación</option>
                    <option value="Sistemas Informáticos">Sistemas Informáticos</option>
                  </select>
                </div>

                {/* Rol */}
                <div className="input-group select-wrapper">
                  <label>Rol del usuario</label>
                  <select
                    value={role}
                    onChange={(e) => setRole(e.target.value)}
                    required
                    className="input-like-select"
                  >
                    <option value="" disabled hidden>
                      Selecciona un rol
                    </option>
                    <option value="Usuario">Usuario</option>
                    <option value="Administrador">Administrador</option>
                  </select>
                </div>

                {/* Contraseña */}
                <div className="input-group">
                  <PasswordField 
                    label="Contraseña" 
                    value={password} 
                    onChange={setPassword} 
                    showPassword={showPassword} 
                    toggleShowPassword={() => setShowPassword((s) => !s)} 
                    autoComplete="new-password" 
                  />
                  
                  {/* Lista de requisitos */}
                  {!allValid && password.length > 0 && (
                    <ul className="text-xs text-red-500 mt-1 list-disc ml-4 space-y-1">
                      {!passwordChecks.length && <li>Entre 8 y 12 caracteres</li>}
                      {!passwordChecks.lowercase && <li>Al menos una minúscula</li>}
                      {!passwordChecks.uppercase && <li>Al menos una mayúscula</li>}
                      {!passwordChecks.number && <li>Al menos un número</li>}
                      {!passwordChecks.symbol && <li>Al menos un símbolo (@$!%*?&)</li>}
                    </ul>
                  )}
                </div>

                {/* Confirmar contraseña */}
                <div className="input-group">
                  <PasswordField 
                    label="Confirmar contraseña" 
                    value={confirmPassword} 
                    onChange={setConfirmPassword} 
                    showPassword={showConfirmPassword} 
                    toggleShowPassword={() => setShowConfirmPassword((s) => !s)} 
                    autoComplete="new-password" 
                  />
                  
                  {confirmPassword.length > 0 && !passwordsMatch && (
                    <p className="!text-xs !text-red-500 !mt-1">
                      Las contraseñas no coinciden
                    </p>
                  )}
                </div>

                <button 
                  type="submit" 
                  className={`btn mt-4 ${!allValid ? "opacity-50 cursor-not-allowed" : ""}`}
                  disabled={!allValid}
                >
                  Registrarse
                </button>
              </form>

              <p className="register-link">
                ¿Ya tienes cuenta?{" "}
                <Link to="/login" className="text-indigo-500 hover:underline">
                  Inicia sesión
                </Link>
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Modal de error */}
      <RegisterErrorModal
        isOpen={showErrorModal}
        onClose={() => setShowErrorModal(false)}
        message={errorMessage}
      />
    </AuthLayout>
  );
}