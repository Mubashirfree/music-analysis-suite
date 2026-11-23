import { useState } from "react";
import UploadPage from "./pages/UploadPage";
import CatalogPage from "./pages/CatalogPage";

const PAGES = {
  Upload: UploadPage,
  Catalogue: CatalogPage,
};
export default function App() {
  const [page, setPage] = useState("Upload");
  const Page = PAGES[page];
  return (
    <>
      <header className="topbar">
        <h1>Music Analysis Suite</h1>
        <nav>
          {Object.keys(PAGES).map((name) => (
            <button
              key={name}
              className={name === page ? "tab active" : "tab"}
              onClick={() => setPage(name)}
            >
              {name}
            </button>
          ))}
        </nav>
      </header>
      <main className="content">
        <Page />
      </main>
    </>
  );
}