import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from './context/AuthContext';
import { AuthContext } from './context/auth-context';
import Login from './pages/Login';
import DataScoringPage from './pages/DataScoringPage';
import FinetuningPage from './pages/FinetuningPage';

const PrivateRoute = ({ children, roles }) => {
    const { user, token } = React.useContext(AuthContext);

    if (!token) {
        return <Navigate to="/login" />;
    }

    if (roles && user && !roles.includes(user.role)) {
        return <Navigate to="/" />; // Redirect if unauthorized
    }

    return children;
};

const DefaultRoute = () => {
    const { user } = React.useContext(AuthContext);
    if (user?.role === 'ROLE_ADMIN') return <Navigate to="/admin/data" replace />;
    if (user?.role === 'ROLE_STUDENT') return <Navigate to="/student/data" replace />;
    return <Navigate to="/login" />;
};

function App() {
  return (
    <Router>
        <AuthProvider>
            <div className="min-h-screen">
                <Routes>
                    <Route path="/login" element={<Login />} />
                    
                    <Route path="/admin/data" element={
                        <PrivateRoute roles={['ROLE_ADMIN']}>
                            <DataScoringPage admin />
                        </PrivateRoute>
                    } />
                    <Route path="/admin/finetune" element={
                        <PrivateRoute roles={['ROLE_ADMIN']}>
                            <FinetuningPage admin />
                        </PrivateRoute>
                    } />
                    <Route path="/admin/*" element={<Navigate to="/admin/data" replace />} />

                    <Route path="/student/data" element={
                        <PrivateRoute roles={['ROLE_STUDENT']}>
                            <DataScoringPage />
                        </PrivateRoute>
                    } />
                    <Route path="/student/finetune" element={
                        <PrivateRoute roles={['ROLE_STUDENT']}>
                            <FinetuningPage />
                        </PrivateRoute>
                    } />
                    <Route path="/student/*" element={<Navigate to="/student/data" replace />} />
                    
                    <Route path="/" element={<DefaultRoute />} />
                </Routes>
            </div>
        </AuthProvider>
    </Router>
  )
}

export default App;
