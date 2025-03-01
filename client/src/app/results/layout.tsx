import "../globals.css";

export const metadata = {
  title: "Fungus Classification Result",
  description: "View the classified fungus species and its details.",
};

export default function ResultsLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="font-roboto bg-gradient-to-r from-green-600 to-green-300 text-white min-h-screen flex items-center justify-center">
      {children}
    </div>
  );
}
