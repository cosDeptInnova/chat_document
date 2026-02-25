import React, { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import AuthLayout from "../layouts/AuthLayout";
import "../styles/login.css";
import PasswordField from "../components/utils/PasswordField";
import RegisterErrorModal from "../components/utils/RegisterErrorModal";
import cosmosLogo from "../images/cosmosLogo.png";

export default function Login({ setUser }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);

  // Estados para manejar el modal de error
  const [showErrorModal, setShowErrorModal] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  const navigate = useNavigate();

  const handleSubmit = (e) => {
    e.preventDefault();

    // Usuario simulado (aquí luego hay que meter el backend o el auth real) con la persistencia del navegador
    const storedUser = JSON.parse(localStorage.getItem("registeredUser"));

    // Al conectar a la bbdd, eliminariamos la persistendia en el navegador en login y register y habria que 
    // enviar en el handleSubmit email y password al backend, validar los credenciales en el backend y después
    // si son correctas, devolver un token o sesión que ahí ya si se guardaría en localstorage (persistencia en el navegador) (todas estas funciones seguramente ya las tenga Rubén desarrolladas)
    if (!storedUser || storedUser.email !== email || storedUser.password !== password) {
        setErrorMessage("Contraseña y/o correo electrónico incorrectos");
        setShowErrorModal(true);
        return;
    }

    // Aqui tambien habria que validar la contraseña aunque se supone que eso lo hará el backend

    // Simulamos autenticación (de momento no verificamos la contraseña)
    localStorage.setItem("user", JSON.stringify(storedUser));
    setUser(storedUser);

    // Redirigir al panel de casos de uso
    navigate("/", { replace: true });
  };

  return (
    <AuthLayout>
        <div className="login-wrapper">
            <div className="container relative z-10 flex">
                <div className="left-panel">
                    <div className="logo">
                        <img
                            src={cosmosLogo}
                            alt="Cosmos Logo"
                            style={{
                            width: "200px",
                            height: "60px",
                            objectFit: "contain",
                            marginRight: "10px",
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

                <div className="form-container">
                    <h2>Iniciar Sesión</h2>
                    <p>Accede a tu cuenta para continuar</p>
                    <form onSubmit={handleSubmit}>
                    <div className="input-group">
                        <label>
                        <i className="fas fa-envelope"></i> Correo electrónico
                        </label>
                        <input
                        type="email"
                        placeholder="tucorreo@cosgs.com"
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                        required
                        />
                    </div>

                    {/* Contraseña usando PasswordField.jsx */}
                    <PasswordField
                        label="Contraseña"
                        value={password}
                        onChange={setPassword}
                        showPassword={showPassword}
                        toggleShowPassword={() => setShowPassword((prev) => !prev)}
                        autoComplete="current-password"
                    />

                    <div className="forgot-password">
                        <a href="#">¿Olvidaste tu contraseña?</a>
                    </div>

                    <button type="submit" className="btn">Iniciar Sesión</button>
                    </form>

                    <p className="text-center text-gray-700">
                        ¿No tienes cuenta?{" "}
                        <Link to="/register" className="text-indigo-500 hover:underline">
                            Regístrate ahora
                        </Link>
                    </p>
                </div>
            </div>
        </div>
        {/* Modal de error en login (reutilizado del registro pero adaptado) */}
        <RegisterErrorModal
            isOpen={showErrorModal}
            onClose={() => setShowErrorModal(false)}
            message={errorMessage}
        />
    </AuthLayout>
    );
}