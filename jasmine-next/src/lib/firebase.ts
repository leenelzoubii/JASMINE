import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";
import { getFirestore } from "firebase/firestore";

const firebaseConfig = {
  apiKey: "AIzaSyDQ1trSa5rCJXZMr6xnvmNhyLBRvIfQL_k",
  authDomain: "jasmine-4671c.firebaseapp.com",
  projectId: "jasmine-4671c",
  storageBucket: "jasmine-4671c.firebasestorage.app",
  messagingSenderId: "33143973243",
  appId: "1:33143973243:web:ed672634ce80895a7ac9de",
  measurementId: "G-X2DKRS1PMP",
};

const app = initializeApp(firebaseConfig);

const auth = getAuth(app);
const db = getFirestore(app);

export { auth, db };