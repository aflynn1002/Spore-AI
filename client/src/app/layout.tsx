import "./globals.css";

export const metadata = {
  title: "Spore AI",
  description: "Discover and classify mushrooms with AI.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        {/* Stronger restrictions on zooming */}
        <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no, viewport-fit=cover" />
      </head>
      <body className="font-roboto bg-gradient-to-r from-green-600 to-green-300 text-white">
        {children}
      </body>
    </html>
  );
}
