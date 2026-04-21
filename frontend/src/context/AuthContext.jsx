/**
 * Authentication Context
 * ======================
 * Provides authentication state and methods across the app.
 */

import React, {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback
} from "react";

// Backend base URL.
// Fall back to relative paths so local Vite proxy works when VITE_API_URL is unset.
const API_URL = (import.meta.env.VITE_API_URL || "").replace(/\/$/, "");

// Auth Context
const AuthContext = createContext(null);

// Custom hook to use auth context
export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
};

// Auth Provider Component
export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(localStorage.getItem("token"));
  const [loading, setLoading] = useState(true);

  const isAuthenticated = Boolean(token && user);
  const isAdmin = user?.role === "admin";

  // 🔹 Validate token on app load
  useEffect(() => {
    const validateToken = async () => {
      if (!token) {
        setLoading(false);
        return;
      }

      try {
        const response = await fetch(`${API_URL}/auth/me`, {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });

        if (response.ok) {
          const userData = await response.json();
          setUser(userData);
        } else {
          localStorage.removeItem("token");
          setToken(null);
          setUser(null);
        }
      } catch (error) {
        console.error("Token validation failed:", error);
        localStorage.removeItem("token");
        setToken(null);
        setUser(null);
      } finally {
        setLoading(false);
      }
    };

    validateToken();
  }, [token]);

  // 🔹 Login
  const login = useCallback(async (username, password) => {
    const response = await fetch(`${API_URL}/auth/login`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ username, password }),
    });

    if (!response.ok) {
      if (response.status === 409) {
        throw new Error("Already logged in on another device");
      }

      const text = await response.text();
      throw new Error(text || "Login failed");
    }

    const data = await response.json();

    localStorage.setItem("token", data.access_token);
    setToken(data.access_token);
    setUser(data.user);

    return data;
  }, []);

  // 🔹 Logout
  const logout = useCallback(async () => {
    try {
      if (token) {
        await fetch(`${API_URL}/auth/logout`, {
          method: "POST",
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });
      }
    } catch (error) {
      console.error("Logout error:", error);
    } finally {
      localStorage.removeItem("token");
      setToken(null);
      setUser(null);
    }
  }, [token]);

  // 🔹 Create user (Admin)
  const createUser = useCallback(
    async (userData) => {
      const response = await fetch(`${API_URL}/auth/register`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(userData),
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || "Failed to create user");
      }

      return await response.json();
    },
    [token]
  );

  // 🔹 Get users (Admin)
  const getUsers = useCallback(async () => {
    const response = await fetch(`${API_URL}/auth/users`, {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || "Failed to fetch users");
    }

    return await response.json();
  }, [token]);

  // 🔹 Delete user (Admin)
  const deleteUser = useCallback(
    async (username) => {
      const response = await fetch(
        `${API_URL}/auth/users/${username}`,
        {
          method: "DELETE",
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );

      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || "Failed to delete user");
      }

      return await response.json();
    },
    [token]
  );

  // 🔹 Toggle user active status (Admin)
  const toggleUserActive = useCallback(
    async (username) => {
      const response = await fetch(
        `${API_URL}/auth/users/${username}/toggle-active`,
        {
          method: "PATCH",
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );

      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || "Failed to toggle user status");
      }

      return await response.json();
    },
    [token]
  );

  const value = {
    user,
    token,
    loading,
    isAuthenticated,
    isAdmin,
    login,
    logout,
    createUser,
    getUsers,
    deleteUser,
    toggleUserActive,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};

export default AuthContext;
