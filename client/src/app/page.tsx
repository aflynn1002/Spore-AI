"use client";

import { useState, useRef } from "react";
import { useRouter } from "next/navigation";

export default function Home() {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [fadeTransition, setFadeTransition] = useState(false);
  const router = useRouter(); // Next.js router instance

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    event.persist();
  
    if (!event.target.files || event.target.files.length === 0) {
      console.warn("No file selected.");
      return;
    }
  
    const file = event.target.files[0];
    const formData = new FormData();
    formData.append("file", file);
  
    try {
      console.log("Uploading file:", file.name);
  
      const response = await fetch("http://localhost:8080/classify", {
        method: "POST",
        body: formData,
      });
  
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Failed to classify file: ${errorText}`);
      }
  
      const data = await response.json();
      console.log("Classification result:", data);
  
      // Store API response in localStorage
      localStorage.setItem("fungusInfo", JSON.stringify(data));
  
      router.push(`/results`);
    } catch (error) {
      console.error("Error uploading file:", error);
      alert(`Classification failed: ${error instanceof Error ? error.message : "An unknown error occurred."}`);
    }
  };
  

  return (
    <div
      className={`flex flex-col md:flex-row items-center justify-center min-h-screen px-6 md:px-16 transition-opacity duration-700 ${
        fadeTransition ? "opacity-0" : "opacity-100"
      }`}
    >
      {/* Mushroom Logo */}
      <img
        src="/mushroom_logo.webp"
        alt="Mushroom Logo"
        className="mushroom-logo w-60 h-60 md:w-96 md:h-96 object-cover mb-6 md:mb-0"
      />

      {/* Content Section */}
      <div className="md:ml-12 flex flex-col items-center md:items-center text-center w-full md:w-1/2">
        <h1 className="spore-title text-4xl md:text-5xl font-bold mb-4">
          Spore AI
        </h1>
        <p className="intro-text max-w-md text-lg md:text-xl leading-relaxed">
          Discover and classify mushrooms with advanced AI technology.
        </p>

        {/* Hidden File Input */}
        <input
          type="file"
          id="file-upload"
          accept="image/*"
          onChange={handleFileChange}
          ref={fileInputRef}
          className="hidden"
        />

        {/* Styled Upload Button (Triggers File Input) */}
        <label
          htmlFor="file-upload"
          className="upload-button mt-6 cursor-pointer bg-green-600 text-white py-3 px-6 rounded-lg text-lg font-medium transition hover:bg-green-700"
        >
          Choose and Upload File
        </label>
      </div>
    </div>
  );
}