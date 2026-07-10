import React, { useState, useContext } from 'react';
import { AuthContext } from '../context/AuthContext';
import { useNavigate } from 'react-router-dom';

export default function Login() {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const { login } = useContext(AuthContext);
    const navigate = useNavigate();

    const handleSubmit = async (e) => {
        e.preventDefault();
        try {
            const res = await fetch('/api/auth/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password })
            });
            let data;
            try {
                data = await res.json();
            } catch (e) {
                data = null;
            }
            
            if (res.ok && data) {
                login(data.token);
                navigate('/');
            } else {
                alert('Login failed: ' + (data?.message || 'Invalid credentials'));
            }
        } catch (err) {
            console.error(err);
            alert('Error occurred during login');
        }
    };

    return (
        <div className="flex items-center justify-center min-h-screen">
            <div className="p-8 bg-white shadow-md rounded w-96">
                <h2 className="text-2xl font-bold mb-6 text-center">Login</h2>
                <form onSubmit={handleSubmit}>
                    <div className="mb-4">
                        <label className="block text-sm font-medium mb-1">Email</label>
                        <input type="email" value={email} onChange={e => setEmail(e.target.value)}
                            className="w-full border p-2 rounded" required />
                    </div>
                    <div className="mb-6">
                        <label className="block text-sm font-medium mb-1">Password</label>
                        <input type="password" value={password} onChange={e => setPassword(e.target.value)}
                            className="w-full border p-2 rounded" required />
                    </div>
                    <button type="submit" className="w-full bg-blue-600 text-white p-2 rounded hover:bg-blue-700">
                        Sign In
                    </button>
                </form>
            </div>
        </div>
    );
}
