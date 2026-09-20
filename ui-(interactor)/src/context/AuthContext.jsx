import { useEffect, useMemo, useState } from 'react';
import { jwtDecode } from 'jwt-decode';
import { useNavigate } from 'react-router-dom';
import { AuthContext } from './auth-context';

function decodeUser(token) {
    if (!token) return null;
    try {
        const decoded = jwtDecode(token);
        if (!decoded.exp || decoded.exp * 1000 <= Date.now()) {
            localStorage.removeItem('fearless_gpus_token');
            return null;
        }
        return { email: decoded.sub, role: decoded.role || 'ROLE_STUDENT' };
    } catch {
        localStorage.removeItem('fearless_gpus_token');
        return null;
    }
}

export const AuthProvider = ({ children }) => {
    const storedToken = localStorage.getItem('fearless_gpus_token');
    const [token, setToken] = useState(() => decodeUser(storedToken) ? storedToken : null);
    const navigate = useNavigate();
    const user = useMemo(() => decodeUser(token), [token]);

    const login = (jwtToken) => {
        localStorage.setItem('fearless_gpus_token', jwtToken);
        setToken(jwtToken);
    };

    const logout = () => {
        localStorage.removeItem('fearless_gpus_token');
        setToken(null);
        navigate('/login');
    };

    useEffect(() => {
        const expireSession = () => {
            localStorage.removeItem('fearless_gpus_token');
            setToken(null);
            navigate('/login');
        };
        window.addEventListener('fearless-gpus:unauthorized', expireSession);
        return () => window.removeEventListener('fearless-gpus:unauthorized', expireSession);
    }, [navigate]);

    return (
        <AuthContext.Provider value={{ user, token, login, logout }}>
            {children}
        </AuthContext.Provider>
    );
};
