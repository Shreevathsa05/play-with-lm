import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, AuthContext } from './context/AuthContext';
import Login from './pages/Login';
import AdminDashboard from './pages/AdminDashboard';
import StudentDashboard from './pages/StudentDashboard';

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
    if (user?.role === 'ROLE_ADMIN') return <Navigate to="/admin" />;
    if (user?.role === 'ROLE_STUDENT') return <Navigate to="/student" />;
    return <Navigate to="/login" />;
};

function App() {
  return (
    <Router>
        <AuthProvider>
            <div className='min-h-screen bg-gray-50 text-gray-900'>
                <Routes>
                    <Route path="/login" element={<Login />} />
                    
                    <Route path="/admin/*" element={
                        <PrivateRoute roles={['ROLE_ADMIN']}>
                            <AdminDashboard />
                        </PrivateRoute>
                    } />
                    
                    <Route path="/student/*" element={
                        <PrivateRoute roles={['ROLE_STUDENT']}>
                            <StudentDashboard />
                        </PrivateRoute>
                    } />
                    
                    <Route path="/" element={<DefaultRoute />} />
                </Routes>
            </div>
        </AuthProvider>
    </Router>
  )
}

export default App;