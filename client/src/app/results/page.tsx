"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { motion } from "framer-motion"; // Import Framer Motion
import { FaInfoCircle, FaTree, FaCheck, FaExclamationCircle, FaLeaf, FaList } from "react-icons/fa";

export default function Results() {
  const [isLoading, setIsLoading] = useState(true);
  const [fungusData, setFungusData] = useState<{
    species: string;
    fungus_info: {
      summary: string;
      habitat: string;
      edibility: string;
      genus: string;
      characteristics: string[];
    };
  } | null>(null);

  useEffect(() => {
    // Simulate a loading delay before displaying results
    setTimeout(() => {
      const storedData = localStorage.getItem("fungusInfo");
      if (storedData) {
        setFungusData(JSON.parse(storedData));
      }
      setIsLoading(false); // Hide loading screen after data is fetched
    }, 2000); // Simulating a 2-second delay
  }, []);

  // Show loading screen while fetching data
  if (isLoading) {
    return (
      <div 
        className="flex flex-col items-center justify-center fixed top-0 left-0 w-screen h-screen"
        style={{ background: "var(--background)", color: "var(--foreground)" }} // Uses your global theme colors
      >
        <div className="loader"></div> {/* Loading Animation */}
        <p className="mt-4 text-lg font-medium">Analyzing your mushroom...</p>
      </div>
    );
  }
  
  

  if (!fungusData) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 1 }}
        className="flex flex-col items-center text-center w-full max-w-3xl px-6 py-8 text-gray-900 rounded-lg shadow-lg bg-white"
      >
        <h1 className="text-3xl font-bold mb-4">No Data Available</h1>
        <p>Please classify an image first.</p>
        <Link
          href="/"
          className="upload-btn mt-6 inline-block bg-green-600 text-white py-3 px-6 rounded-lg text-lg font-medium transition hover:bg-green-700"
        >
          Classify Another
        </Link>
      </motion.div>
    );
  }

  const { species, fungus_info } = fungusData;
  const isEdible = fungus_info.edibility.toLowerCase().trim() === "edible";

  // Format species name to match file structure
  const formattedSpecies = species.replace(/\s+/g, "_").toLowerCase(); // Example: "Amanita muscaria" → "amanita_muscaria"
  const imageUrl = `/baymushroomphotos/${formattedSpecies}.jpg`; // Assuming images are in JPG format

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }} // Start position
      animate={{ opacity: 1, y: 0 }} // End position
      transition={{ duration: 1 }} // Animation duration
      className="flex flex-col items-center text-center w-full max-w-3xl px-6 py-8 text-gray-900 rounded-lg shadow-lg bg-white"
    >
      <h1 className="text-3xl font-bold mb-4">Classification Result</h1>
      <p className="text-lg font-medium">
        <strong>Species:</strong> {species}
      </p>

      {/* Display Mushroom Image */}
      <img
        src={imageUrl}
        alt={`Image of ${species}`}
        className="w-full max-w-md h-auto rounded-lg shadow-md mt-4"
        onError={(e) => {
          (e.target as HTMLImageElement).src = "/placeholder.png"; // Fallback image if not found
        }}
      />

      {/* Widgets Section */}
      <div className="widgets grid grid-cols-1 md:grid-cols-2 gap-4 w-full mt-6">
        <div className="widget result-widget p-4 rounded-lg shadow bg-white">
          <h2 className="text-xl font-semibold mb-2 flex items-center justify-center">
            <FaInfoCircle className="mr-2 text-blue-500" />
            Summary
          </h2>
          <p>{fungus_info.summary}</p>
        </div>
        <div className="widget result-widget p-4 rounded-lg shadow bg-white">
          <h2 className="text-xl font-semibold mb-2 flex items-center justify-center">
            <FaTree className="mr-2 text-green-500" />
            Habitat
          </h2>
          <p>{fungus_info.habitat}</p>
        </div>
        <div className="widget result-widget p-4 rounded-lg shadow bg-white">
          <h2 className="text-xl font-semibold mb-2 flex items-center justify-center">
            {isEdible ? (
              <FaCheck className="mr-2 text-green-600" />
            ) : (
              <FaExclamationCircle  className="mr-2 text-gray-500" />
            )}
            Edibility
          </h2>
          <p>{fungus_info.edibility}</p>
        </div>
        <div className="widget result-widget p-4 rounded-lg shadow bg-white">
          <h2 className="text-xl font-semibold mb-2 flex items-center justify-center">
            <FaLeaf className="mr-2 text-purple-500" />
            Genus
          </h2>
          <p>{fungus_info.genus}</p>
        </div>
        <div className="widget result-widget p-4 rounded-lg shadow md:col-span-2 bg-white">
          <h2 className="text-xl font-semibold mb-2 flex items-center justify-center">
            <FaList className="mr-2 text-orange-500" />
            Characteristics
          </h2>
          <ul className="list-disc list-inside text-left">
            {fungus_info.characteristics.length > 0 ? (
              fungus_info.characteristics.map((char, index) => (
                <li key={index} className="text-gray-800">
                  {char}
                </li>
              ))
            ) : (
              <p>No characteristics available.</p>
            )}
          </ul>
        </div>
      </div>

      {/* Return Button */}
      <Link
        href="/"
        className="upload-btn mt-6 inline-block bg-green-600 text-white py-3 px-6 rounded-lg text-lg font-medium transition hover:bg-green-700"
      >
        Classify Another
      </Link>
    </motion.div>
  );
}
