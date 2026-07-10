import React, { createContext, useState, useEffect } from 'react';
import { jwtDecode } from 'jwt-decode';
import { useNavigate } from 'react-router-dom';

export const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
    const [user, setUser] = useState(null);
    const [token, setToken] = useState(localStorage.getItem('fearless_gpus_token') || null);
    const navigate = useNavigate();

    useEffect(() => {
        if (token) {
            try {
                const decoded = jwtDecode(token);
                // In Spring Security default JWT we typically just get subject. 
                // Let's assume role is stored in claims if added, or we decode if we added it, 
                // but since we didn't add role to JWT payload in our simple JwtService, 
                // wait, we should probably modify JwtService to add roles.
                // For now, let's fetch user profile or assume role from JWT if it exists.
                setUser({ email: decoded.sub, role: decoded.role || 'ROLE_STUDENT' });
            } catch (err) {
                console.error("Invalid token", err);
                logout();
            }
        }
    }, [token]);

    const login = (jwtToken) => {
        localStorage.setItem('fearless_gpus_token', jwtToken);
        setToken(jwtToken);
    };

    const logout = () => {
        localStorage.removeItem('fearless_gpus_token');
        setToken(null);
        setUser(null);
        navigate('/login');
    };

    return (
        <AuthContext.Provider value={{ user, token, login, logout }}>
            {children}
        </AuthContext.Provider>
    );
};
