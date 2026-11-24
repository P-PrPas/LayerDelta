import {createContext, useContext, useEffect, useInsertionEffect, useState, type ReactNode} from "react";

type Preference = "system" | "dark" | "light";
type Theme = Exclude<Preference, "system">;
const storageKey = "layerdelta.theme";
const system = window.matchMedia("(prefers-color-scheme: dark)");
const ThemeContext = createContext({theme: "dark" as Theme, preference: "system" as Preference,
  setPreference: (_value: Preference) => {}});

function readPreference(): Preference {
  try {
    const saved = localStorage.getItem(storageKey);
    if (saved === "dark" || saved === "light") return saved;
  } catch { /* Settings still work when browser storage is unavailable. */ }
  return "system";
}

export function ThemeProvider({children}: {children: ReactNode}) {
  const [preference, setPreference] = useState(readPreference);
  const [systemDark, setSystemDark] = useState(system.matches);
  const theme = preference === "system" ? systemDark ? "dark" : "light" : preference;
  useInsertionEffect(() => { document.documentElement.dataset.theme = theme; }, [theme]);
  useEffect(() => {
    const change = () => setSystemDark(system.matches);
    system.addEventListener("change", change);
    return () => system.removeEventListener("change", change);
  }, []);
  return <ThemeContext.Provider value={{theme, preference, setPreference(value) {
    setPreference(value);
    try { localStorage.setItem(storageKey, value); } catch { /* Keep this window's preference. */ }
  }}}>{children}</ThemeContext.Provider>;
}

export function useTheme() {
  return useContext(ThemeContext);
}

export function ThemeSetting() {
  const {preference, setPreference} = useTheme();
  return <label className="theme-setting">
    Theme
    <select value={preference} onChange={e => setPreference(e.target.value as Preference)}>
      <option value="system">System</option>
      <option value="dark">Dark</option>
      <option value="light">Light</option>
    </select>
  </label>;
}
